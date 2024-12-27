import os
import re
import copy
import random
from collections import defaultdict

import torch
from tqdm import tqdm
from glob import glob

from .base_runner import BaseRunner, console
from utils.etc_utils import load_jsonl, load_txt, load_json


class SummarizerRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)
        
        self.args = args

        self.save_dir = os.path.join(self.output_base_dir, 'dialogue-summary', f'persona_seed:{args.persona_seed_num}')
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.last_save_chunk_idx_file = os.path.join(self.save_dir, 'last_save_chunk_idx.txt')

        self._load_prompt_template()

    def _load_prompt_template(self):
        self.template = load_txt('./templates/dialogue_summary.txt')

    @property
    def system_msg(self):
        return "Your job is to summarize the given conversation."

    @property
    def prompt_prefix(self):
        #if self.args.image_alignment_target == 'mobile-device-image':
        return "dialogue-summary"

    def convert_flatten_dialogue(self, dialogue):
        
        flatten_dialogue = []
        for instance in dialogue:
            print(instance)
            spk = instance['speaker']
            utter = instance['utterance']

            if len(instance['sharing_info']) != 0:
                image_desc = instance['sharing_info']['image_description']
                flatten_dialogue.append(f'{spk}: [Sharing Image of {image_desc}]')
            else:
                flatten_dialogue.append(f'{spk}: {utter}')
        
        return '\n'.join(flatten_dialogue)

    def prepare_prompt(self):
        try:
            results = load_jsonl(os.path.join(self.output_base_dir, 'dialogue', f'persona_seed:{self.args.persona_seed_num}', f'session_num:{self.args.target_session_num}', 'final_output.jsonl'))
            console.log('[{}] # of Total results: {}'.format(self.__class__.__name__, len(results)))
        except FileNotFoundError as e:
            return []

        if self.args.debug:
            try:
                results = random.sample(results, self.args.debug_sample_num)
            except ValueError as e:
                results = results

        prompts = []
        for instance in tqdm(results, total=len(results)):
            print(instance.keys())
            print(instance['dialogue:date'])
            current_date = instance['dialogue:date']
            name = instance['name']

            print(instance['dialogue:last_date'])
            print(instance['dialogue:history_event'])
            print(instance['session_number'])
            flatten_dialogue = self.convert_flatten_dialogue(instance['parsed_dialogue_generation'])
            
            prompt = self.template.format(current_date=current_date, dialogue=flatten_dialogue, name=name)
            print(prompt)
            assert False

    def parse_and_filter(self):
        return None