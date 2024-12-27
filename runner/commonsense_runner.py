import os
import re
import copy
import random
import glob
from tqdm import tqdm
from collections import defaultdict

from .base_runner import BaseRunner, console
from utils.etc_utils import load_jsonl, load_txt, load_json
from utils.persona_utils import (
    AGE_LIST,
    GENDER_LIST,
    COUNTRY_LIST,
    ETHNICITY_LIST,
    RELIGION_LIST,
    COUNTRY_ALPHA_LIST,
    EDUCATION_LIST,
    EXCLUDE_COMMONSENSE_TARGET,
    PERSONA_CATEGORY_LIST,
    DEMOGRAPHIC_TARGET,
    COMMONSENSE_TARGET,
    PEACOK_RELATION
)

DEMOGRAPHIC_TRIAL = 5


class CommonsenseRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)

        self.save_dir = os.path.join(self.output_base_dir, 'commonsense', f'persona_seed:{args.persona_seed_num}')
        os.makedirs(self.save_dir, exist_ok=True)
        self.last_save_chunk_idx_file = os.path.join(self.save_dir, 'last_save_chunk_idx.txt')

        self._load_peacok_prompt_template()

    @property
    def system_msg(self):
        return "You are a helpful assistant."

    @property
    def prompt_prefix(self):
        return "commonsense"

    def get_system_message(self, relation):
        _sys_msg_map = {
            'characteristic': "Your task is to infer a character trait of the person in the given sentence, something that sets them apart from others.",
            'experience': "Your task is to infer what the person in the given sentence has done in the past, something that sets them apart from others.", 
            'goal': "Your task is to infer what the person in the given sentence will do or achieve in the future, something that sets them apart from others.", 
            'relationship': "Your task is to infer a relationship that the person in the given sentence has with other people or social groups, something that sets them apart from others.", 
            'routine': "Your task is to infer what the person in the given sentence regularly or consistently does, something that sets them apart from others."
        }
        return _sys_msg_map[relation]

    def _load_peacok_prompt_template(self):
        self.peacok_template = dict()
        
        for relation in PEACOK_RELATION:
            self.peacok_template[relation] = self.load_prompt_template(f"commonsense-{relation}")
    

    def _create_demographic_sentence(self):
        age_range = random.sample(AGE_LIST, 1)[0]
        age = random.randint(int(age_range.split('-')[0]), int(age_range.split('-')[1]))
        gender = random.sample(GENDER_LIST, 1)[0]
        country = random.sample(COUNTRY_LIST, 1)[0]
        #ethnicity = random.sample(ETHNICITY_LIST, 1)[0]
        #education = random.sample(EDUCATION_LIST, 1)[0]
        return {
            'age': age,
            'gender': gender,
            'nationality': country
        }


    def prepare_prompt(self):
        
        persona_sentence = load_jsonl(os.path.join(self.output_base_dir, 'persona-attr', f'final_output_{self.args.persona_seed_num}.jsonl'))
        console.log('[{}] # of Total persona sentence: {}'.format(self.__class__.__name__, len(persona_sentence)))

        if self.args.debug:
            persona_sentence = random.sample(persona_sentence, self.args.debug_sample_num)

        prompts = []
        for instance in tqdm(persona_sentence, total=len(persona_sentence)):
            age = instance['age']
            gender = instance['gender']
            #nationality = instance['nationality']
            birthplace = instance['birthplace']
            residence = instance['residence']

            for relation, template in self.peacok_template.items():
                demo_sent = "I am a {}-year-old {}. I was born in {}, I currently reside in {}.".format(
                    age,
                    gender.lower(),
                    birthplace, residence
                )

                _prompt = template.format(
                    demo_sent=demo_sent,
                    persona_attr=instance['persona-attr:sent']
                )
                
                cp_instance = copy.deepcopy(instance)
                cp_instance[f'{self.prompt_prefix}_prompt'] = _prompt
                cp_instance[f'{self.prompt_prefix}_system_message'] = self.get_system_message(relation)
                cp_instance[f'{self.prompt_prefix}_relation'] = relation

                prompts.append(cp_instance)
        
        return prompts

    def _load_prompt_template_for_parsing(self, relation):
        return {
            'routine': '{demo_sent} {persona_attr} I regularly {commonsense}.',
            'characteristic': '{demo_sent} {persona_attr} I {commonsense}.',
            'experience': 'I {commonsense}. Now, {demo_sent} {persona_attr}',
            'goal': '{demo_sent} {persona_attr} I plan {commonsense}.',
            'relationship': '{demo_sent} {persona_attr} So, I {commonsense}.'
        }
    
    def _parse_commonsense_generation(self, demo_sent, persona_attr, generation, relation):
        template = {
            'routine': '{demo_sent} {persona_attr} I regularly ',
            'characteristic': '{demo_sent} {persona_attr} I ',
            'experience': ' Now, {demo_sent} {persona_attr}',
            'goal': '{demo_sent} {persona_attr} I plan ',
            'relationship': '{demo_sent} {persona_attr} So, I '
        }
        
        parse_sent = template[relation].format(demo_sent=demo_sent, persona_attr=persona_attr)
        
        if parse_sent in generation:
            parsed_commonsense = generation.replace(parse_sent, '')
            if parsed_commonsense[-1] == '.':
                parsed_commonsense = parsed_commonsense[:-1]

            if relation == 'experience':
                parsed_commonsense = parsed_commonsense.split('I ')[-1]
            
            return parsed_commonsense
        
        if relation == 'characteristic':
            parsed_commonsense = self._parse_characteristic_generation(generation, persona_attr)
        elif relation == 'experience':
            parsed_commonsense = self._parse_experience_generation(generation, persona_attr)
        elif relation == 'goal':
            parsed_commonsense = self._parse_goal_generation(generation, persona_attr)
        elif relation == 'relationship':
            parsed_commonsense = self._parse_relationship_generation(generation, persona_attr)
        elif relation == 'routine':
            parsed_commonsense = self._parse_routine_generation(generation, persona_attr)
        else:
            raise ValueError('Wrong relation type name!')

        if not parsed_commonsense:
            parsed_commonsense = generation

        if parsed_commonsense[-1] == '.':
            parsed_commonsense = parsed_commonsense[:-1]

        return parsed_commonsense

    def _parse_characteristic_generation(self, generation, persona_attr):
        #return generation.replace(f'{persona_attr} I ', '')
        parsed_commonsense = None

        if 'I ' in generation and generation.startswith('I '):
            parsed_commonsense = generation.split('I ')[-1]
        
        return parsed_commonsense

    def _parse_experience_generation(self, generation, persona_attr):
        parsed_commonsense = None
        if ' Now, ' in generation:
            parsed_commonsense = generation.split(' Now, ')[0]

            if 'I ' in generation and generation.startswith('I '):
                parsed_commonsense = parsed_commonsense.split('I ')[-1]
        
        elif 'I ' in generation and generation.startswith('I '):
            parsed_commonsense = generation.split('I ')[-1]
        
        return parsed_commonsense

    def _parse_goal_generation(self, generation, persona_attr):
        parsed_commonsense = None
        if 'I plan ' in generation:
            parsed_commonsense = generation.split('I plan ')[-1]
        
        elif 'I ' in generation and generation.startswith('I '):
            parsed_commonsense = generation.split('I ')[-1]

        return parsed_commonsense

    def _parse_relationship_generation(self, generation, persona_attr):
        parsed_commonsense = None
        if 'So, I ' in generation:
            parsed_commonsense = generation.split('So, I ')[-1]
        
        elif 'I ' in generation and generation.startswith('I '):
            parsed_commonsense = generation.split('I ')[-1]

        return parsed_commonsense

    def _parse_routine_generation(self, generation, persona_attr):
        parsed_commonsense = None
        if 'I regularly ' in generation:
            parsed_commonsense = generation.split('I regularly ')[-1]
        
        elif 'I ' in generation and generation.startswith('I '):
            parsed_commonsense = generation.split('I ')[-1]

        return parsed_commonsense

    def parse_and_filter(self, generations):
        '''
        Remove repetitions
        '''
        unique_results = []
        stat = defaultdict(int)
        seen = set()
        
        for generation in tqdm(generations, total=len(generations)):
            
            persona_sent = generation['persona-attr:sent']
            relation = generation[f'{self.prompt_prefix}_relation']
            commonsense = generation[f'{self.prompt_prefix}_generation']

            age = generation['age']
            gender = generation['gender']
            #nationality = generation['nationality']
            birthplace = generation['birthplace']
            residence = generation['residence']

            demo_sent = "I am a {}-year-old {}. I was born in {}, I currently reside in {}.".format(
                age,   
                gender.lower(),
                #nationality,
                birthplace, residence
            )
            
            parsed_commonsense = self._parse_commonsense_generation(demo_sent, persona_sent, commonsense, relation)
            
            key = '{}+++{}+++{}'.format(persona_sent, relation, parsed_commonsense)

            if key not in seen:
                seen.add(key)

                cp_generation = copy.deepcopy(generation)
                cp_generation[f'parsed_{self.prompt_prefix}_generation'] = parsed_commonsense
                unique_results.append(cp_generation)

        stat['commonsense:total'] = len(generations)
        stat['commonsense:success'] = len(unique_results)

        self.dump_output(unique_results, os.path.join(self.save_dir, 'final_output.jsonl'))
        self.dump_report(stat, os.path.join(self.save_dir, 'report_output.txt'))
        
    def _construct_persona_sentence_w_demographic(self):
        demographic_attributes = dict()
        # get one sentence from each demographic information
        for category in DEMOGRAPHIC_TARGET:
            if isinstance(category, list):
                random.shuffle(category)
                selected_category = category[0]

                _persona_attr = load_jsonl(os.path.join(self.output_base_dir, 'persona-attr', f'{selected_category}-success-output.jsonl'))
                random.shuffle(_persona_attr)

                demographic_attributes[selected_category] = _persona_attr[0]
                continue

            _persona_attr = load_jsonl(os.path.join(self.output_base_dir, 'persona-attr', f'{category}-success-output.jsonl'))
            random.shuffle(_persona_attr)
            demographic_attributes[category] = _persona_attr[0]
        
        console.log(demographic_attributes, style='debugging')

        return demographic_attributes
    
    def _generate_commonsense_knowledge(self, prompts, prompt_prefix=None):
        return self.interact(prompts, prompt_prefix)