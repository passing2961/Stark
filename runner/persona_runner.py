import os
import copy
import re
import random
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base_runner import BaseRunner, console
from utils.persona_utils import (
    PERSONA_CATEGORY_MAP,
    PERSONA_CATEGORY_LIST,
    AGE_LIST,
    GENDER_LIST,
    COUNTRY_LIST,
    COUNTRY_ALPHA_LIST,
    COUNTRY_NAME2ALPHA,
    ETHNICITY_LIST,
    RELIGION_LIST,
    EDUCATION_LIST,
    EXCLUDE_COMMONSENSE_TARGET
)
from utils.etc_utils import load_json, load_jsonl


NUM_TRIALS = 20
THRESHOLD = 0.8

DEMOGRAPHIC_TRIAL = 10

SYSTEM_MESSAGE = """Based on the given persona category, entity key, and user's profile information (i.e., age, gender, nationality), your job is to generate 30 persona sentences and corresponding persona entity values in the format "<persona sentence> (<entity key>: <entity value>)." You should generate very specific persona sentences and entity values. The persona sentence can express a positive sentiment (like) or a negative one (dislike). 

For example, 

{example}"""


class LiteDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        premise = item['persona-attr:sent']
        hypothesis = 'This sentence is about {}.'.format(item['persona-attr:key'].lower())

        tokenized_ids = self.tokenizer.encode(
            premise, 
            hypothesis, 
            return_tensors='pt',
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        
        cp_item = copy.deepcopy(item)
        cp_item['input_tensor'] = tokenized_ids[0]

        return cp_item

    def collate(self, batch):
        keys = set([key for b in batch for key in b.keys()])

        dict_batch = defaultdict(list)
        for key in keys:
            for b in batch:
                if key != 'input_tensor':
                    dict_batch[key].append(b[key])

        dict_batch['input_tensor'] = torch.stack([b['input_tensor'] for b in batch], dim=0)

        return dict_batch

class PersonaRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)

        #self.persona_attr_prompt = 'Given the persona category and entity key, you should generate 20 persona sentences and persona entity values in the format "<persona sentence> (<entity key>: <entity value>)." You should generate very specific persona sentences and entity values. The persona sentence can express a positive sentiment (like) or a negative one (dislike).\n\nFor example,\n\n{few_shot_example}\n\nPersona Category: {target_persona_category}\nPersona Entity Key: {target_persona_entity}\nPersona Sentences:\n1.'
        self.save_dir = os.path.join(self.output_base_dir, self.prompt_prefix)
        os.makedirs(self.save_dir, exist_ok=True)
        self.last_save_chunk_idx_file = os.path.join(self.save_dir, 'last_save_chunk_idx.txt')

        self.template = self.load_prompt_template("persona")
        self._load_persona_attribute()

    @property
    def system_msg(self):
        return "You are a helpful assistant."

    @property
    def prompt_prefix(self):
        return "persona-attr"

    def _load_persona_attribute(self):
        
        results = defaultdict(list)
        for filename, instance in PERSONA_CATEGORY_MAP.items():

            _persona_attr_set = load_json(f'./datasets/ProfileGen/{filename}.json')
            
            # process the loaded persona attribute sets
            # because certain cases contain irrelevant <attr> information compared to the map
            # for example, entity key should be "food", but the loaded attr is "preference".
            _persona_entity_key = instance[1]
            if _persona_entity_key == 'movie_title':
                _persona_entity_key = 'movie title'
            _persona_attr_set = self._remove_irrelevant_persona_attr_sample(_persona_attr_set, _persona_entity_key)

            results['{}+++{}'.format(instance[0], _persona_entity_key)].extend(_persona_attr_set)

        self.persona_attribute_set = results
        self.persona_attribute_key = list(results.keys())

    def _create_demographic_sentence(self, do_sep_sample=False):
        age_range = random.sample(AGE_LIST, 1)[0]
        age = random.randint(int(age_range.split('-')[0]), int(age_range.split('-')[1]))
        gender = random.sample(GENDER_LIST, 1)[0]
        #country = random.sample(COUNTRY_LIST, 1)[0]
        birthplace = random.sample(COUNTRY_LIST, 1)[0]
        
        if do_sep_sample:
            country = random.sample(COUNTRY_LIST, 2)
            birthplace = country[0]
            residence = country[1]
            #residence = random.sample(COUNTRY_LIST, 1)[0]
        else:
            residence = birthplace
        #ethnicity = random.sample(ETHNICITY_LIST, 1)[0]
        #education = random.sample(EDUCATION_LIST, 1)[0]
        return {
            'age': age,
            'gender': gender,
            #'nationality': country
            'residence': residence,
            'birthplace': birthplace
        }

    def get_system_message(self, example):
        return SYSTEM_MESSAGE.format(example=example)

    def prepare_prompt(self):
        prompts = []
        for instance in PERSONA_CATEGORY_LIST:
            target_persona_category = instance[0]
            target_persona_entity_key = instance[1]
            if target_persona_category in EXCLUDE_COMMONSENSE_TARGET:
                continue
            #if target_persona_category != 'Preference ⊃ Drink':
            #    continue
            
            # we first collect all inputs with few-shot demonstrations
            # each target category contains 5 times few-shot demonstrations
            # thus, we will obtain 100 (20*5) generations for each category.

            for idx in range(DEMOGRAPHIC_TRIAL):
                if idx > 0.7 * DEMOGRAPHIC_TRIAL:
                    _demographic_info = self._create_demographic_sentence(True)
                else:
                    _demographic_info = self._create_demographic_sentence()

                system_prompts = self._collect_persona_attribute(target_persona_category=target_persona_category, target_persona_entity_key=target_persona_entity_key)

                age = _demographic_info['age']
                gender = _demographic_info['gender']
                #nationality = _demographic_info['nationality']
                birthplace = _demographic_info['birthplace']
                residence = _demographic_info['residence']

                _prompt = self.template.format(
                    age=age, 
                    gender=gender, 
                    #nationality=nationality,
                    birthplace=birthplace,
                    residence=residence,
                    target_persona_category=target_persona_category,
                    target_persona_entity=target_persona_entity_key
                )
                
                for system_prompt in system_prompts:
                    prompts.append({
                        'persona_category': target_persona_category,
                        'persona_entity_key': target_persona_entity_key,
                        f'{self.prompt_prefix}_system_message': self.get_system_message(system_prompt),
                        f'{self.prompt_prefix}_prompt': _prompt,
                        'age': age,
                        'gender': gender,
                        'birthplace': birthplace,
                        'residence': residence,
                        'birthplace_alpha2_code': COUNTRY_NAME2ALPHA[birthplace],
                        'residence_alpha2_code': COUNTRY_NAME2ALPHA[residence],
                        #'nationality': nationality
                    })

        console.log('[{}] # of target persona categories: {}'.format(self.__class__.__name__, len(PERSONA_CATEGORY_LIST)), style='info')
        console.log('[{}] List of target persona category: {}'.format(self.__class__.__name__, PERSONA_CATEGORY_LIST), style='info')
        
        return prompts

    def prepare_prompt_for_demo(self, kwargs):
        prompts = []
        
        target_persona_category = kwargs['target_persona_category']
        target_persona_entity_key = kwargs['target_persona_entity']
        
        system_prompts = self._collect_persona_attribute(target_persona_category=target_persona_category, target_persona_entity_key=target_persona_entity_key)
    #prompts[target_persona_category] = _input_prompt #.extend(_input_prompt)

        age = kwargs['age']
        gender = kwargs['gender']
        birthplace = kwargs['birthplace']
        residence = kwargs['residence']

        _prompt = self.template.format(**kwargs)

        for system_prompt in system_prompts[:1]:
            prompts.append({
                'persona_category': target_persona_category,
                'persona_entity_key': target_persona_entity_key,
                f'{self.prompt_prefix}_system_message': self.get_system_message(system_prompt),
                f'{self.prompt_prefix}_prompt': _prompt,
                'age': age,
                'gender': gender,
                'nationality': nationality
            })

        
        return prompts
    
    def _collect_persona_attribute(
        self, 
        target_persona_category: str = None,
        target_persona_entity_key: str = None
    ):

        input_prompts = []
        for _ in range(NUM_TRIALS):
            random.shuffle(self.persona_attribute_key)
            few_shot_persona_attribute_key = self.persona_attribute_key[:1]

            few_shot_persona_attribute_prompt = self._construct_few_shot_persona_attribute_prompt(few_shot_persona_attribute_key)

            #persona_attribute_prompt = self.template.format(
            #    few_shot_example=few_shot_persona_attribute_prompt,
            #    target_persona_category=target_persona_category,
            #    target_persona_entity=target_persona_entity_key
            #)
            
            #input_prompts.append(persona_attribute_prompt)
            input_prompts.append(few_shot_persona_attribute_prompt)
        
        return input_prompts

    def _construct_few_shot_persona_attribute_prompt(self, few_shot_key):
        
        few_shot_prompt = []
        few_shot_prompt_template = 'Persona Category: {persona_category}\nPersona Entity Key: {persona_entity_key}\nPersona Sentences:\n{persona_sentences}' # {persona_sentence} ({persona_entity_key}: {persona_entity_value})'
        for _key in few_shot_key:
            _few_shot_instances = self.persona_attribute_set[_key]
            random.shuffle(_few_shot_instances)
            _few_shot_samples = _few_shot_instances[:30]

            # Double-check the consistency about the entity key
            _few_shot_example_prompt = []
            for i, _few_shot_sample in enumerate(_few_shot_samples):
                assert _key.split('+++')[1].title() == _few_shot_sample['attr'].title()

                _few_shot_example_prompt.append(
                    '{}. {} ({}: {})'.format(i+1, _few_shot_sample['sentence'], _key.split('+++')[1].title(), _few_shot_sample['value'].title()))

            _prompt = few_shot_prompt_template.format(
                persona_category=_key.split('+++')[0],
                persona_entity_key=_key.split('+++')[1].title(),
                persona_sentences='\n'.join(_few_shot_example_prompt)
                #persona_sentence=_few_shot_sample['sentence'],
                #persona_entity_value=_few_shot_sample['value']
            )
            
            few_shot_prompt.append(_prompt)
        
        return '\n\n'.join(few_shot_prompt)

    def parse_and_filter(self, generations):

        stat = defaultdict(int)
        stat['total_num'] = len(generations) * 30

        regex_parsed_results, regex_discard_results = [], []
        for generation in tqdm(generations, total=len(generations)):
            
            parsed_results, discard_results = self._parse_persona_attribute_generation(generation[f'{self.prompt_prefix}_generation'])

            for parsed_result in parsed_results:
                cp_generation = copy.deepcopy(generation)
                uuid = self.generate_unique_uuid()
                cp_generation['id'] = uuid
                for k, v in parsed_result.items():
                    cp_generation[f'{self.prompt_prefix}:{k}'] = parsed_result[k]
                regex_parsed_results.append(cp_generation)
            
            for discard_result in discard_results:
                cp_generation = copy.deepcopy(generation)
                cp_generation['regex:discard_result'] = discard_result
                regex_discard_results.append(cp_generation)
        
        stat['regex:parsed_result'] = len(regex_parsed_results)
        stat['regex:discard_result'] = len(regex_discard_results)

        self.dump_output(regex_parsed_results, os.path.join(self.save_dir, 'regex_parsed_output.jsonl'))
        self.dump_output(regex_discard_results, os.path.join(self.save_dir, 'regex_discard_output.jsonl'))

        exact_parsed_results, exact_discard_results = [], []
        for regex_parsed_result in tqdm(regex_parsed_results, total=len(regex_parsed_results)):
            
            target_entity_key = regex_parsed_result['persona_entity_key']
            parsed_sent = regex_parsed_result[f'{self.prompt_prefix}:sent']
            parsed_key = regex_parsed_result[f'{self.prompt_prefix}:key']
            parsed_value = regex_parsed_result[f'{self.prompt_prefix}:value']

            if parsed_key.lower() == target_entity_key and parsed_value.lower() in parsed_sent.lower():
                exact_parsed_results.append(regex_parsed_result)
            else:
                exact_discard_results.append(regex_parsed_result)
        
        stat['exact:parsed_result'] = len(exact_parsed_results)
        stat['exact:discard_result'] = len(exact_discard_results)

        self.dump_output(exact_parsed_results, os.path.join(self.save_dir, 'exact_parsed_output.jsonl'))
        self.dump_output(exact_discard_results, os.path.join(self.save_dir, 'exact_discard_output.jsonl'))

        unique_results = []
        seen = set()
        for exact_parsed_result in tqdm(exact_parsed_results, total=len(exact_parsed_results)):
            key = exact_parsed_result[f'{self.prompt_prefix}:sent']
            
            if key not in seen:
                seen.add(key)
                unique_results.append(exact_parsed_result)

        stat['duplication:unique_result'] = len(unique_results)

        self.dump_output(unique_results, os.path.join(self.save_dir, 'unique_output.jsonl'))

        unique_results = load_jsonl(os.path.join(self.save_dir, 'unique_output.jsonl'))
        
        # make seed dataset
        random.shuffle(unique_results)
        
        buffer_count = 0
        buffer_index = 0
        buffer_list = []

        for unique_result in tqdm(unique_results, total=len(unique_results)):
            buffer_list.append(unique_result)

            buffer_count += 1
            if buffer_count == self.buffer_size:
                self.dump_output(buffer_list, os.path.join(self.save_dir, f'final_output_{buffer_index}.jsonl'))
                buffer_list = []
                buffer_count = 0
                buffer_index += 1

        self.dump_output(buffer_list, os.path.join(self.save_dir, f'final_output_{buffer_index}.jsonl'))
        self.dump_report(stat, os.path.join(self.save_dir, 'report_output.txt'))

    def _remove_irrelevant_persona_attr_sample(self, persona_attr_set, persona_entity_key):
        result = []
        for item in persona_attr_set:
            if item['attr'].lower() != persona_entity_key.lower():
                continue
            result.append(item)
        return result

    def _parse_persona_attribute_generation(self, generation):
        """
        We extract the useful persona-related information from the structured format (i.e., <sent> (<key>: <value>))
        using the regex pattern. Any sentence which doesn't match with the regex pattern, we regard the sentence as an inappropriate generated sentence.
        """
        
        # First, split the generation based on the number prefix (e.g., 1., 2.)
        delims = [f'\n{i}. ' for i in range(1, 31)] + [f'\n{i}.' for i in range(1, 31)]

        splitted_generation = re.split('|'.join(delims), generation)
        #if len(splitted_generation) != 30:
        #    splitted_generation = splitted_generation[:30]
        #assert len(splitted_generation) == 20, '{}'.format(generation)

        # Second, extract the persona-related information using the regex pattern
        pattern = '(?P<sent>.*) [\(|\[](?P<key>.*): (?P<value>.*)[\)|\]]' # [] case should be possible
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

    def _generate_single_persona_attribute(self, persona_attribute_prompt):

        llm_generation = self.interact(persona_attribute_prompt)
        
        return llm_generation

    def _generate_persona_attribute(self, prompts, prompt_prefix=None):
        return self.interact(prompts, prompt_prefix)
