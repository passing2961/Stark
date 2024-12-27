from abc import ABC, abstractmethod
import os
import sys
import json
import time
import copy
import uuid
import random
from pathlib import Path

import faiss
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import openai
from openai import OpenAI, AsyncOpenAI
from rich.console import Console
from rich.theme import Theme

from utils.etc_utils import load_jsonl, load_txt
from runner.retrieval import process_image, load_diffusion_model, load_clip_model

custom_theme = Theme({
    "info": "bold dim cyan",
    "warning": "bold magenta",
    "danger": "bold red",
    "debugging": "bold sandy_brown"
})
console = Console(theme=custom_theme)


class BaseRunner(ABC):
    def __init__(self, args):
        self.args = args
        self.output_base_dir = os.path.join('./outputs', self.args.run_id + ":{}".format(self.args.model))
        os.makedirs(self.output_base_dir, exist_ok=True)
        
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        self.buffer_size = 100

        self.existing_uuids = set()

        self.diffusion_model = None
        self.image_db = None
        self.retrieval_model = None
        self.retrieval_preprocess = None
        if "image" in self.args.runner_name:
            if self.args.image_process_type == 'retrieve':
                self.image_db = self.load_image_db()
                self.retrieval_model, self.retrieval_preprocess = load_clip_model()

            elif self.args.image_process_type == 'generate':
                self.diffusion_model = load_diffusion_model(args.diffusion_model_id, args.cache_dir)

    def load_image_db(self):
        
        image_list = dict()
        image_indices = dict()
        for image_dataset in ["coco_train2017", "gqa", "ocr_vqa", "redcaps12m", "textvqa", "vg"]:
            data_dir = Path(f'prepare_image_db/embeddings_folder/{image_dataset}/metadata')
            df = pd.concat(
                pd.read_parquet(parquet_file)
                for parquet_file in data_dir.glob('*.parquet')
            )

            ind = faiss.read_index(f'prepare_image_db/index_folder/{image_dataset}/knn.index')
            image_list[image_dataset] = df['image_path'].tolist()
            image_indices[image_dataset] = ind

        return {
            'image_list': image_list,
            'image_indices': image_indices
        }

    def run(self):
        
        if self.args.do_parse_filter:
            if self.args.runner_name == 'dialogue' and not os.path.isfile(os.path.join(self.save_dir, 'output.jsonl')):
                return
            if self.args.runner_name == 'image-alignment' and not os.path.isfile(os.path.join(self.save_dir, 'output.jsonl')):
                return
            generations = load_jsonl(os.path.join(self.save_dir, 'output.jsonl'))
            self.parse_and_filter(generations)
            #sys.exit()
        else:
            self.generate()

    @abstractmethod
    def prepare_prompt(self):
        """
        Prepare prompts for generations
        """
        pass
    
    def generate(self):
        """
        Prompting LLM via call OpenAI API
        """
        
        if 'image' in self.args.runner_name and self.args.runner_name != 'image-alignment':
            prompts, image_prompts = self.prepare_prompt()
            
            processed_output = process_image(
                image_prompts, 
                retrieval_model=self.retrieval_model,
                retrieval_preprocess=self.retrieval_preprocess,
                diffusion_model=self.diffusion_model, 
                image_db=self.image_db,
                prompt_prefix=self.prompt_prefix,
                image_process_type=self.args.image_process_type
            )
            if self.args.image_process_type == 'retrieve':
                self.dump_output(processed_output, os.path.join(self.save_dir, 'output.jsonl'))
            elif self.args.image_process_type == 'generate':
                self.dump_output(prompts, os.path.join(self.save_dir, 'output.jsonl'))
            return

        prompts = self.prepare_prompt()

        if self.args.runner_name == 'dialogue' and len(prompts) == 0:
            return
        
        if self.args.runner_name == 'image-alignment' and len(prompts) == 0:
            return
        
        if self.args.runner_name == 'face':
            self.dump_output(prompts, os.path.join(self.save_dir, 'output.jsonl'))
            return

        console.log(prompts[0]['{}_prompt'.format(self.prompt_prefix)], style='info')
        
        if self.args.debug:
            if len(prompts) >= self.args.debug_sample_num:
                prompts = random.sample(prompts, self.args.debug_sample_num)

        console.log('[{}] # of prompt: {}'.format(self.__class__.__name__, len(prompts)), style='info')
        
        last_save_chunk_idx = self._load_last_save_chunk_idx()
        chunk_prompts = self._sharding(prompts, shard_num=self.args.shard_num)
        t = tqdm(total=len(chunk_prompts))
        
        print(last_save_chunk_idx)
        
        for chunk_idx, chunk_prompt in enumerate(chunk_prompts):
            if chunk_idx <= last_save_chunk_idx:
                t.update(1)
                print(chunk_idx, last_save_chunk_idx)
                continue
            
            chunk_generation = self._interact(chunk_prompt, prompt_prefix=self.prompt_prefix)
         
            self.dump_chunk_output(chunk_idx, chunk_generation, os.path.join(self.save_dir, 'output.jsonl'))
            t.update(1)

    @abstractmethod
    def parse_and_filter(self):
        """
        Parsing and Filtering LLM-generated results
        """
        pass

    @property
    def system_msg(self):
        """
        System message for each step
        """
        raise NotImplementedError("Subclasses must implement this property")

    @property
    def prompt_prefix(self):
        """
        Prompt prefix for each step
        """
        raise NotImplementedError("Subclasses must implement this property")

    def _parse_json_list(self, sent):
        sent = sent.replace('```json', '')
        #sent = sent.replace('```', '')
        sent = sent.split('```')[0] # sometimes LLM generates an explanation about their decision after the token ```

        return sent

    def _call_openai_api(self, prompt_input, prompt_prefix=None):
        prompt = prompt_input[f'{prompt_prefix}_prompt']

        if self.args.runner_name == 'commonsense':
            system_message = prompt_input['commonsense_system_message']
        elif self.args.runner_name == 'persona-attr':
            system_message = prompt_input['persona-attr_system_message']
        else:
            system_message = self.system_msg

        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.args.model,
                    messages=[{"role": "system", "content": system_message}, {"role": "user", "content": "{}".format(prompt)}],
                    temperature=self.args.temperature,
                    max_tokens=self.args.max_tokens,
                    top_p=self.args.top_p,
                    frequency_penalty=self.args.frequency_penalty,
                    presence_penalty=self.args.presence_penalty,
                    #stop='\n\n'
                )
                break
            except (RuntimeError, openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError) as e:
                print("Error: {}".format(e))
                time.sleep(2)
                continue
            
        output = completion.choices[0].message.content.strip()
        completion_tokens = completion.usage.completion_tokens
        prompt_tokens = completion.usage.prompt_tokens
        
        prompt_input[f'{prompt_prefix}_generation'] = output
        prompt_input[f'{prompt_prefix}:prompt_tokens'] = prompt_tokens
        prompt_input[f'{prompt_prefix}:completion_tokens'] = completion_tokens
        
        return prompt_input

    def generate_unique_uuid(self):
        while True:
            new_uuid = str(uuid.uuid4())
            if new_uuid not in self.existing_uuids:
                self.existing_uuids.add(new_uuid)
                return new_uuid

    def dump_output(self, outputs, file_name=None):
        f = open(file_name, 'w') 
        for output in outputs:
            f.write(json.dumps(output) + '\n')
        f.close()

    def dump_json_output(self, outputs, file_name=None):
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(outputs, f, ensure_ascii=False, indent='\t')
    
    def dump_chunk_output(self, chunk_idx, outputs, file_name=None):
        # update save point
        with open(self.last_save_chunk_idx_file, 'w') as f:
            f.write(str(chunk_idx))

        f = open(file_name, 'a')
        for output in outputs:
            f.write(json.dumps(output) + '\n')
        f.close()

    def dump_error_message(self, error_msg, file_name=None):
        with open(file_name, 'w') as f:
            json.dump(error_msg, f, ensure_ascii=False, indent='\t')

    def dump_report(self, outputs, file_name=None):
        f = open(file_name, 'w')
        for k, v in outputs.items():
            f.write('{}\t{}\n'.format(k, str(v)))
        f.close()
    
    def _load_last_save_chunk_idx(self):
        if os.path.exists(self.last_save_chunk_idx_file):
            with open(self.last_save_chunk_idx_file, 'r') as f:
                last_save_chunk_idx = int(f.readlines()[0].strip())
        else:
            last_save_chunk_idx = -1
        
        return last_save_chunk_idx
    
    def load_prompt_template(self, template_name: str):
        return load_txt(f'./templates/{template_name}.txt')

    def _sharding(self, input_list, shard_num=None):
        shard_size = len(input_list) // shard_num
        remainder = len(input_list) % shard_num

        shards = []
        for i in range(shard_num):
            start_index = i * shard_size + min(i, remainder)
            end_index = start_index + shard_size + (1 if i < remainder else 0)
            if len(input_list[start_index:end_index]) == 0:
                continue
            shards.append(input_list[start_index:end_index])
        
        return shards

    def _interact(self, prompts, prompt_prefix=None):
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []

            for instance in tqdm(prompts, total=len(prompts)):
                cp_instance = copy.deepcopy(instance)

                future = executor.submit(self._call_openai_api, cp_instance, prompt_prefix=prompt_prefix)
                futures.append(future)

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                ret = future.result()
                results.append(ret)

        return results