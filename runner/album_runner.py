import os
import re
import copy
import random
from tqdm import tqdm
from glob import glob
from collections import defaultdict

from .base_runner import BaseRunner, console
from utils.etc_utils import load_jsonl, load_txt


SYSTEM_MESSAGE = """Given the sentence related to a person's daily life, your task is to generate five image descriptions that could be stored on the person's mobile device, along with corresponding image categories. You should use the format "<image_description> (Category: <image_category>)". The image category may include selfies, past memories, screenshots, landmarks, animals, art, celebrities, nature, and food.

For example, 

My name is Tom. I am a 32-year-old man. I was born in the USA and currently reside there. I have a strong interest in basketball. I played basketball in middle school, but now I work as a chatbot developer at a startup. I enjoy watching the NBA because I love basketball.

Image descriptions stored on Tom's mobile device:
1. A photo of a young Tom playing basketball in a middle school gymnasium (Category: Past Memory, Sport)
2. A selfie of Tom smiling at the Golden State Warriors' arena during a game (Category: Selfie, Sport)
3. A screenshot of chatbot development code using Python (Category: Screenshot, Computer, Software)
4. A picture of Tom enjoying a night out with coworkers at a local pub (Category: Social Networking, Food, Drink)
5. A photo of Tom meeting a famous NBA player at a basketball event (Category: Celebrity, Sport)"""

class AlbumRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)

        self.save_dir = os.path.join(self.output_base_dir, 'mobile-device', f'persona_seed:{args.persona_seed_num}')
        os.makedirs(self.save_dir, exist_ok=True)

        self._load_prompt_template()
        self.last_save_chunk_idx_file = os.path.join(self.save_dir, 'last_save_chunk_idx.txt')

    @property
    def system_msg(self):
        return SYSTEM_MESSAGE

    @property
    def prompt_prefix(self):
        return "mobile-device"

    def _load_prompt_template(self):
        self.album_template = load_txt('./templates/mobile_device.txt')

    def prepare_prompt(self):
        
        event_graph = load_jsonl(os.path.join(self.output_base_dir, 'event-graph', f'persona_seed:{self.args.persona_seed_num}', 'final_output.jsonl'))
        console.log('[{}] # of Total event graph: {}'.format(self.__class__.__name__, len(event_graph)))

        prompts = []
        for instance in tqdm(event_graph, total=len(event_graph)):
            narrative = instance['narrative_generation']
            
            _prompt = self.album_template.format(
                name=instance['name'],
                #age=instance['demographic:age'],
                #gender=instance['demographic:gender'],
                #nationality=instance['demographic:nationality'],
                sentence=instance['narrative_generation']
            )

            cp_instance = copy.deepcopy(instance)
            cp_instance[f'{self.prompt_prefix}_prompt'] = _prompt
            prompts.append(cp_instance)

        return prompts
    
    def parse_and_filter(self, generations):
        
        stat = defaultdict(int)
        stat['total_num'] = len(generations) * 5

        results = []
        regex_parsed_results, regex_discard_results = [], []
        for generation in tqdm(generations, total=len(generations)):
            parsed_results, discard_results = self._parse_mobile_device_generation(generation[f'{self.prompt_prefix}_generation'])
            if len(parsed_results) == 0:
                for discard_result in discard_results:
                    cp_generation = copy.deepcopy(generation)
                    cp_generation['regex:discard_result'] = discard_result
                    regex_discard_results.append(cp_generation)
                continue

            cp_instance = copy.deepcopy(generation)
            cp_instance[f'parsed_{self.prompt_prefix}_generation'] = parsed_results
            results.append(cp_instance)

            for parsed_result in parsed_results:
                cp_generation = copy.deepcopy(generation)
                for k, v in parsed_result.items():
                    cp_generation[f'{self.prompt_prefix}:{k}'] = parsed_result[k]
                regex_parsed_results.append(cp_generation)
            
            for discard_result in discard_results:
                cp_generation = copy.deepcopy(generation)
                cp_generation['regex:discard_result'] = discard_result
                regex_discard_results.append(cp_generation)
        
        stat['regex:parsed_result'] = len(regex_parsed_results)
        stat['regex:discard_result'] = len(regex_discard_results)

        self.dump_output(results, os.path.join(self.save_dir, 'final_output.jsonl'))
        self.dump_output(regex_parsed_results, os.path.join(self.save_dir, 'regex_parsed_output.jsonl'))
        self.dump_output(regex_discard_results, os.path.join(self.save_dir, 'regex_discard_output.jsonl'))
        self.dump_report(stat, os.path.join(self.save_dir, 'report_output.txt'))

    def _parse_mobile_device_generation(self, generation):
        # First, split the generation based on the number prefix (e.g., 1., 2.)
        delims = [f'\n{i}. ' for i in range(1, 6)] + [f'\n{i}.' for i in range(1, 6)]
        splitted_generation = re.split('|'.join(delims), generation)
        
        # Second, extract the persona-related information using the regex pattern
        pattern = '(?P<image_description>.*) [\(|\[]Category: (?P<image_category>.*)[\)|\]]' # [] case should be possible
        compiled_regex = re.compile(pattern)

        parsed_results = []
        discard = []
        for generation in splitted_generation:
            matched = compiled_regex.match(generation)

            if matched:
                parsed_results.append(matched.groupdict())
            else:
                discard.append(generation)

        return parsed_results, discard