import os
import re
import copy
import random
from tqdm import tqdm
from glob import glob

from .base_runner import BaseRunner, console
from utils.etc_utils import load_jsonl, load_txt


SYSTEM_MESSAGE = """Given the profile information, your job is to generate a detailed description of a human that includes specific details such as background, tops, bottoms, hair, and shoes.

For example, 

Profile Information:
- Age: 32
- Gender: Woman
- Nationality: South Korea

Human Description: A full-body shot, an Asian adult female, fit, small road with trees, straight red above-chest hair, normal-length, white and long sleeve cotton shirt, short plaid skirt in pleated shape, cotton backpack, socks, black leather oxford shoes."""

STYLE_TYPES = [
    "full-body", "upper body", "portrait", "headshot", "nearly full-body"
]

HUMAN_ATTR_KEY_ORDER = {
    'body shape': ['body shape'],
    'background': ['background'],
    'hair': ['hair style', 'hair color', 'hair length'],
    'special clothings': ['sleeve length', 'type'],
    'one-piece outfits': ['shoulder exposure level', 'length', 'collar shape', 'sleeve length', 'material', 'pattern', 'type'],
    'tops': ['graphic', 'color', 'collar shape', 'top length', 'sleeve length', 'material', 'pattern', 'type'],
    'coats': ['graphic', 'color', 'collar shape', 'coat length', 'material', 'pattern', 'type'],
    'bottoms': ['graphic', 'color', 'bottom shape', 'length', 'material', 'pattern', 'type'],
    'shoes': ['color', 'boots length', 'material', 'pattern', 'type'],
    'bags': ['material', 'type'],
    'hats': ['material', 'type'],
    'belts': ['material'],
    'scarf': ['material', 'pattern'],
    'headband': ['material', 'pattern'],
    'headscarf': ['material', 'pattern'],
    'veil': ['material', 'pattern'],
    'socks': ['material', 'pattern'],
    'ties': ['material', 'pattern']
}

class FaceRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)

        self.save_dir = os.path.join(self.output_base_dir, 'face', f'persona_seed:{args.persona_seed_num}')
        os.makedirs(self.save_dir, exist_ok=True)

        self._load_human_attribute_pool()
        self._load_prompt_template()
        self.last_save_chunk_idx_file = os.path.join(self.save_dir, 'last_save_chunk_idx.txt')

    @property
    def system_msg(self):
        return SYSTEM_MESSAGE

    @property
    def prompt_prefix(self):
        return "face"

    def _load_prompt_template(self):
        self.face_template = load_txt('./templates/face.txt')

    def _load_human_attribute_pool(self):
        self.human_attribute_pool = load_jsonl(os.path.join('./datasets/cosmic/human_attribute_pool.jsonl'))

    def make_human_attribute_sentence(self, human_attr, age, gender, birthplace):
        style_type = random.sample(STYLE_TYPES, 1)[0]
        gender = gender.lower()

        template = f'A {style_type} shot, a {age}-years-old {gender} from {birthplace},' #-years-old 
        for category, attribute in human_attr.items():
            if category == 'face':
                continue
            if category == 'overall-style':
                continue
            
            if category == 'hair':
                assert 'wears' not in attribute.keys()
            order_keys = HUMAN_ATTR_KEY_ORDER[category]
            for order_key in order_keys:
                try:
                    template += ' {}'.format(attribute[order_key])
                except KeyError as e:
                    continue
            #for k, v in attribute.items():
            #    template += f' {v}'
            
            if category == 'hair':
                template += ' hair'
            
            if category == 'belts':
                template += ' belt'
            
            if category == 'scarf':
                template += ' scarf'

            if category == 'headband':
                template += ' headband'
            
            if category == 'headscarf':
                template += ' headscarf'
            if category == 'veil':
                template += ' veil'
            if category == 'socks':
                template += ' socks'
            if category == 'ties':
                template += ' tie'
            
            template += ','

        template = template[:-1] + '.'
        
        return template

    def prepare_prompt(self):
        persona_sentence = load_jsonl(os.path.join(self.output_base_dir, 'persona-attr', f'final_output_{self.args.persona_seed_num}.jsonl'))
        console.log('[{}] # of Total persona sentence: {}'.format(self.__class__.__name__, len(persona_sentence)))

        if self.args.debug:
            persona_sentence = random.sample(persona_sentence, self.args.debug_sample_num)
        
        sample_num = len(persona_sentence)
        sampled_human_attribute = random.sample(self.human_attribute_pool, sample_num)
        assert len(persona_sentence) == len(sampled_human_attribute)

        prompts = []
        for idx, instance in enumerate(tqdm(persona_sentence, total=len(persona_sentence))):
            human_attr = sampled_human_attribute[idx]
            
            human_attr_sent = self.make_human_attribute_sentence(
                human_attr,
                instance['age'], instance['gender'], instance['birthplace']
            )

            cp_instance = copy.deepcopy(instance)
            cp_instance[f'{self.prompt_prefix}_prompt'] = human_attr_sent
            
            prompts.append(cp_instance)
        
        return prompts

    def parse_and_filter(self, generations):
        self.dump_output(generations, os.path.join(self.save_dir, 'final_output.jsonl'))
    
    def _generate_album(self, prompts, prompt_prefix=None):
        return self.interact(prompts, prompt_prefix=prompt_prefix)