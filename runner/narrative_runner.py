import os
import copy
import random
from tqdm import tqdm
from glob import glob
from collections import defaultdict

from names_dataset import NameDataset

from .base_runner import BaseRunner, console
from utils.etc_utils import load_jsonl, load_txt
from utils.persona_utils import COUNTRY_ALPHA_LIST


class NarrativeRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)
        
        self.save_dir = os.path.join(self.output_base_dir, 'narrative', f'persona_seed:{args.persona_seed_num}')
        os.makedirs(self.save_dir, exist_ok=True)

        self.nd = NameDataset()

        self._load_prompt_template()
        self._load_universe_name_list()
        self.last_save_chunk_idx_file = os.path.join(self.save_dir, 'last_save_chunk_idx.txt')

    @property
    def system_msg(self):
        return "You are a helpful assistant."

    @property
    def prompt_prefix(self):
        return "narrative"
        
    def _load_persona_commonsense_knowledge(self):
        our_peacok = load_jsonl(os.path.join(self.output_base_dir, 'commonsense-knowledge', 'final_output_0.jsonl'))
        
        self.persona_CK = our_peacok
        console.log('[{}] Done Loading Persona Commonsense Knowledge..'.format(self.__class__.__name__))

    def _load_universe_name_list(self):
        all_names = defaultdict(dict)
        for country_alpha2_code in COUNTRY_ALPHA_LIST:
            top_names = self.nd.get_top_names(n=1000, country_alpha2=country_alpha2_code)

            male_names = top_names[country_alpha2_code]['M']
            female_names = top_names[country_alpha2_code]['F']

            all_names[country_alpha2_code] = {
                'Male': male_names,
                'Female': female_names,
                'Non-binary': male_names + female_names
            }

        self.name_group = all_names

    def _load_prompt_template(self):
        self.sentence_form_template = {
            'routine': 'My name is {name}. {demo_sent} {persona_attr} I regularly {commonsense}.',
            'characteristic': 'My name is {name}. {demo_sent} {persona_attr} I {commonsense}.',
            'experience': 'My name is {name}. I {commonsense}. Now, {demo_sent} {persona_attr}',
            'goal': 'My name is {name}. {demo_sent} {persona_attr} I plan {commonsense}.',
            'relationship': 'My name is {name}. {demo_sent} {persona_attr} So, I {commonsense}.'
        }
        
        self.sentence_to_narrative_template = load_txt('./templates/narrative.txt')
    
    def prepare_prompt(self):
        
        persona_CK = load_jsonl(os.path.join(self.output_base_dir, 'commonsense', f'persona_seed:{self.args.persona_seed_num}', 'final_output.jsonl'))
        console.log('[{}] # of Total persona commonsense: {}'.format(self.__class__.__name__, len(persona_CK)))

        if self.args.debug:
            persona_CK = random.sample(persona_CK, self.args.debug_sample_num)
        
        prompts = []
        for ck in tqdm(persona_CK, total=len(persona_CK)):
        
            persona_attr = ck['persona-attr:sent']
            commonsense = ck['parsed_commonsense_generation']
            relation = ck['commonsense_relation']

            age = ck['age']
            gender = ck['gender']
            #nationality = ck['nationality']
            birthplace = ck['birthplace']
            residence = ck['residence']

            birthplace_alpha2_code = ck['birthplace_alpha2_code']
            sampled_name = random.sample(self.name_group[birthplace_alpha2_code][gender], 1)[0]
            
            demo_sent = "I am a {}-year-old {}. I was born in {}, I currently reside in {}.".format(
                #sampled_name,
                age,
                gender.lower(),
                birthplace, residence
            )

            sentence_form = self.sentence_form_template[relation].format(
                demo_sent=demo_sent,
                persona_attr=persona_attr, 
                commonsense=commonsense,
                name=sampled_name
            )
            
            prompt = self.sentence_to_narrative_template.format(
                sentence=f'{sentence_form}',
            )
            
            cp_instance = copy.deepcopy(ck)
            cp_instance[f'{self.prompt_prefix}_sentence_form'] = sentence_form
            cp_instance[f'{self.prompt_prefix}_prompt'] = prompt
            cp_instance['name'] = sampled_name
            
            prompts.append(cp_instance)
        
        return prompts
    
    def parse_and_filter(self, generations):
        self.dump_output(generations, os.path.join(self.save_dir, 'final_output.jsonl'))

    def _generate_initial_narrative(self, prompts, prompt_prefix=None):
        return self.interact(prompts, prompt_prefix=prompt_prefix)