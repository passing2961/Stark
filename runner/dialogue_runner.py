import os
import sys
import copy
import random
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

from .base_runner import BaseRunner, console
from utils.etc_utils import load_jsonl, load_txt

current_year = datetime.now().year


SYSTEM_MESSAGE = """Your job is to generate a long in-depth conversation between an user and an user-friendly AI assistant with multiple turns. The user and AI assistant can share images during a conversation in order to strengthen social relationship, to convey important information, to amuse/entertain, to clarify complex situations, to change the topic of dialogue, or to express emotions/opinions/reactions. There must be more than two image-sharing moments within the conversation. The shared images can either be from the collection previously stored on the user's mobile device or obtained from the internet. You must generate the conversation following the guidelines below.

[Guideline]
- The conversation is represented in the form of a json list.
- Each entry is a python dictionary containing the following keys: "utterance_id", "speaker", "utterance", "sharing_info".
- The "utterance_id" field contains a unique identifier for the utterance within the conversation.
- The "speaker" field contains a speaker of the utterance.
- The "utterance" field contains the utterance of the speaker. If the image-sharing behavior occurs, then the "utterance" is a empty string.
- The "sharing_info" field represents the image-sharing moment and is represented in the form of a python dictionary containing the following keys: "rationale", "image_description", "image_source", "keywords", "image_id_from_mobile".
- If the image-sharing moment does not occur, then the "sharing_info" field is an empty python dictionary.
- The "rationale" field represents the reason behind sharing the relevant image.
- The "image_description" field contains a description of the shared image.
- The "image_source" field contains a source of the shared image whether it is from the internet (internet) or the user's mobile device (mobile).
- If you select the user's mobile device as the "image_source," you must either share an image that matches one of the existing descriptions already on the user's mobile device or share a new image that does not exist among these descriptions.
- If you share an image that matches one of the existing descriptions on the user's mobile device, you must generate the appropriate image ID in the "image_id_from_mobile" field.
- If you share a new image that does not match any existing descriptions on the user's mobile device, you must enter "new added image" in the "image_id_from_mobile" field.
- The "keywords" field contains keywords of the shared image."""

class DialogueRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)
        
        self.save_dir = os.path.join(self.output_base_dir, 'dialogue', f'persona_seed:{args.persona_seed_num}', f'session_num:{args.target_session_num}')
        os.makedirs(self.save_dir, exist_ok=True)
        self.last_save_chunk_idx_file = os.path.join(self.save_dir, 'last_save_chunk_idx.txt')

        self._load_prompt_template()
    
    @property
    def system_msg(self):
        return SYSTEM_MESSAGE

    @property
    def prompt_prefix(self):
        return "dialogue"

    def _load_temporal_event_graph(self):
        return load_jsonl(os.path.join(self.output_base_dir, 'event-graph', 'final_output.jsonl'))
    
    def _load_prompt_template(self):
        self.dialogue_template = {
            'first_session': load_txt('./templates/first_stark_dialogue.txt'),
            'next_session': load_txt('./templates/next_stark_dialogue.txt')    
        }

    def _parse_json_list(self, sent):
        sent = sent.replace('```json', '')
        sent = sent.replace('```', '')
        
        return sent
    
    def _extract_target_session_input(self, event_graph):
        max_session_num = max([len(instance['parsed_event-graph_generation']) for instance in event_graph])

    def prepare_prompt(self):
        if self.args.target_session_num == 1:
            temporal_event_graph = load_jsonl(os.path.join(self.output_base_dir, 'mobile-device', f'persona_seed:{self.args.persona_seed_num}', 'final_output.jsonl'))
        else:
            if not os.path.isfile(os.path.join(self.output_base_dir, 'dialogue', f'persona_seed:{self.args.persona_seed_num}', f'session_num:{self.args.target_session_num-1}', 'final_output.jsonl')):
                return []

            temporal_event_graph = load_jsonl(os.path.join(self.output_base_dir, 'dialogue', f'persona_seed:{self.args.persona_seed_num}', f'session_num:{self.args.target_session_num-1}', 'final_output.jsonl'))
        console.log('[{}] # of Total temporal event graph: {}'.format(self.__class__.__name__, len(temporal_event_graph)))

        max_session_num = max([len(instance['parsed_event-graph_generation']) for instance in temporal_event_graph])
        if max_session_num < self.args.target_session_num:
            return []

        prompts = []
        for idx, instance in tqdm(enumerate(temporal_event_graph)):
            if len(instance['parsed_event-graph_generation']) < self.args.target_session_num:
                continue

            session_prompt = self._construct_session_prompt(instance)
            assert len(session_prompt) == 1

            cp_instance = copy.deepcopy(instance)
            cp_instance.update(session_prompt[0])
            prompts.append(cp_instance)
            
        return prompts

    def prepare_prompt_all_session(self):
        temporal_event_graph = load_jsonl(os.path.join(self.output_base_dir, 'mobile-device', f'persona_seed:{self.args.persona_seed_num}', 'final_output.jsonl'))
        console.log('[{}] # of Total temporal event graph: {}'.format(self.__class__.__name__, len(temporal_event_graph)))

        self._extract_target_session_input(temporal_event_graph)
        
        if self.args.debug:
            temporal_event_graph = random.sample(temporal_event_graph, self.args.debug_sample_num)
        
        prompts = []
        for idx, instance in tqdm(enumerate(temporal_event_graph)):
            _session_prompts = self._construct_session_prompt(instance)
            dialogue_uuid = self.generate_unique_uuid()

            for _session_prompt in _session_prompts:
                cp_instance = copy.deepcopy(instance)
                cp_instance['dialogue_uuid'] = dialogue_uuid
                cp_instance.update(_session_prompt)
                prompts.append(cp_instance)
            
        return prompts
    
    def _construct_session_prompt(self, instance):

        event_graph = instance['parsed_event-graph_generation']
        name = instance['name']
        age = instance['age']
        gender = instance['gender']
        #nationality = instance['nationality']
        birthplace = instance['birthplace']
        residence = instance['residence']

        slot_info = {
            'name': name,
            'age': age,
            'gender': gender,
            'birthplace': birthplace,
            'residence': residence,
            'mobile_device': [{'image_id': idx, 'image_description': ele['image_description']} for idx, ele in enumerate(instance['parsed_mobile-device_generation'])],
            'history_event': []
        }

        prompts = []
        for event_id, sub_event in enumerate(event_graph):
            

            if event_id + 1 != int(sub_event['id']): # 'Mismatch ID'
                console.log('event_id: {}, sub_event_id: {}'.format(event_id+1, sub_event))
                #break
            
            if event_id == 0:
                slot_info['event'] = sub_event['event']
                slot_info['date'] = sub_event['date']
                if self.args.target_session_num == event_id + 1:
                    _prompt = self._construct_first_session_prompt(**slot_info)
            else:
                slot_info['event'] = sub_event['event']
                slot_info['date'] = sub_event['date']
                
                if 'caused_by' in sub_event:
                    
                    if not sub_event['caused_by']:
                        assert False
                        _prompt = self._construct_first_session_prompt(**slot_info)
                    else:
                        slot_info['time_interval'] = sub_event['caused_by']['caused_by:time_interval']
                        slot_info['experience'] = sub_event['caused_by']['caused_by:experience']
                        
                        if self.args.target_session_num == event_id + 1:
                            _prompt = self._construct_next_session_prompt(**slot_info)

            
            #slot_info['last_event'] = sub_event['event']
            slot_info['last_date'] = sub_event['date']
            slot_info['history_event'].append('- {}: {}'.format(sub_event['date'], sub_event['event']))

            _slot_info = {f'{self.prompt_prefix}:{slot_k}': slot_v for slot_k, slot_v in slot_info.items()}
            if event_id + 1 == self.args.target_session_num:
                _slot_info[f'{self.prompt_prefix}_prompt'] = _prompt
                _slot_info['session_number'] = event_id + 1
                prompts.append(_slot_info)
                break
            
        return prompts

    def _construct_session_prompt_all(self, instance):

        event_graph = instance['parsed_event-graph_generation']
        name = instance['name']
        age = instance['age']
        gender = instance['gender']
        #nationality = instance['nationality']
        birthplace = instance['birthplace']
        residence = instance['residence']

        slot_info = {
            'name': name,
            'age': age,
            'gender': gender,
            'birthplace': birthplace,
            'residence': residence,
            'mobile_device': [{'image_id': idx, 'image_description': ele['image_description']} for idx, ele in enumerate(instance['parsed_mobile-device_generation'])],
            'history_event': []
        }

        prompts = []
        for event_id, sub_event in enumerate(event_graph):
            if event_id + 1 != int(sub_event['id']): # 'Mismatch ID'
                console.log('event_id: {}, sub_event_id: {}'.format(event_id+1, sub_event))
                #break
            
            if event_id == 0:
                slot_info['event'] = sub_event['event']
                slot_info['date'] = sub_event['date']
                _prompt = self._construct_first_session_prompt(**slot_info)
            else:
                slot_info['event'] = sub_event['event']
                slot_info['date'] = sub_event['date']
                
                if 'caused_by' in sub_event:
                    
                    if not sub_event['caused_by']:
                        assert False
                        _prompt = self._construct_first_session_prompt(**slot_info)
                    else:
                        slot_info['time_interval'] = sub_event['caused_by']['caused_by:time_interval']
                        slot_info['experience'] = sub_event['caused_by']['caused_by:experience']
                        _prompt = self._construct_next_session_prompt(**slot_info)

            #slot_info['last_event'] = sub_event['event']
            slot_info['last_date'] = sub_event['date']
            slot_info['history_event'].append('- {}: {}'.format(sub_event['date'], sub_event['event']))

            _slot_info = {f'{self.prompt_prefix}:{slot_k}': slot_v for slot_k, slot_v in slot_info.items()}
            _slot_info[f'{self.prompt_prefix}_prompt'] = _prompt
            _slot_info['session_number'] = event_id + 1
            prompts.append(_slot_info)

        return prompts

    def _construct_first_session_prompt(self, **kwargs):
        return self.dialogue_template['first_session'].format(**kwargs)
    
    def _construct_next_session_prompt(self, **kwargs):
        kwargs['history_event'] = '\n'.join(kwargs['history_event'])
        return self.dialogue_template['next_session'].format(**kwargs)

    def check_json_key_exist(self, parsed_event):
        for instance in parsed_event:
            
            for key in ['utterance_id', 'speaker', 'utterance', 'sharing_info']:
                if key not in instance.keys():
                    return False

            if len(instance['sharing_info']) != 0:
                
                if isinstance(instance['sharing_info'], list):
                    print(instance['sharing_info'])
                    #assert len(instance['sharing_info']) == 1
                    instance['sharing_info'] = copy.deepcopy(instance['sharing_info']) #[0]
                
                    for sharing_instance in instance['sharing_info']:
                        for key in ['rationale', 'image_description', 'image_source', 'keywords']: #, "image_id_from_mobile"]:
                            if key not in sharing_instance.keys():
                                return False
                elif isinstance(instance['sharing_info'], dict):
                    for key in ['rationale', 'image_description', 'image_source', 'keywords']: #, "image_id_from_mobile"]:
                        if key not in instance['sharing_info'].keys():
                            return False

        return True
    
    def extract_new_mobile_device_image(self, parsed_dialogue, mobile_device_pool):
        new_image_desc = []
        for instance in parsed_dialogue:
            if len(instance['sharing_info']) == 0:
                continue
            
            if isinstance(instance['sharing_info'], list):
                for item in instance['sharing_info']:
                    try:
                        if item['image_id_from_mobile'] == 'new added image' and item['image_source'] == 'mobile':
                            new_image_desc.append({'image_description': item['image_description']})
                    except KeyError as e:
                        continue
            elif isinstance(instance['sharing_info'], dict):

                try:
                    if instance['sharing_info']['image_id_from_mobile'] == 'new added image' and instance['sharing_info']['image_source'] == 'mobile':
                        new_image_desc.append({'image_description': instance['sharing_info']['image_description']})
                except KeyError as e:
                    continue
                
        new_mobile_device_pool = mobile_device_pool + new_image_desc

        return new_mobile_device_pool

    def parse_and_filter(self, generations):
        results, fail_results = [], []
        stat = defaultdict(int)
        error = []

        for generation in tqdm(generations, total=len(generations)):
            
            try:
                parsed_dialogue = eval(self._parse_json_list(generation[f'{self.prompt_prefix}_generation']))
            except (SyntaxError, TypeError, NameError) as e:
                console.log(self._parse_json_list(generation[f'{self.prompt_prefix}_generation']), style='warning')
                stat[f'{self.prompt_prefix}_syntax:fail'] += 1
                error.append(str(e))
                continue

            # if the type of parsed result is a python dictionary or the length of parsed result is 1,
            # then we eliminate these results regarded as low-quality samples
            if isinstance(parsed_dialogue, dict) or len(parsed_dialogue) == 1:
                stat[f'{self.prompt_prefix}_parse:fail'] += 1
                continue

            if not self.check_json_key_exist(parsed_dialogue):
                stat[f'{self.prompt_prefix}_key_existence:fail'] += 1
                fail_results.append(parsed_dialogue)
                continue
            
            new_added_image = self.extract_new_mobile_device_image(parsed_dialogue, generation['parsed_mobile-device_generation'])
            
            cp_generation = copy.deepcopy(generation)
            cp_generation[f'parsed_{self.prompt_prefix}_generation'] = parsed_dialogue
            cp_generation[f'parsed_mobile-device_generation'] = new_added_image
            results.append(cp_generation)

        stat['dialogue:total'] = len(generation)

        self.dump_output(results, os.path.join(self.save_dir, 'final_output.jsonl'))
        self.dump_report(stat, os.path.join(self.save_dir, 'report.txt'))
        self.dump_error_message(error, os.path.join(self.save_dir, 'error_message.json'))
        self.dump_json_output(fail_results, os.path.join(self.save_dir, 'fail_output.json'))

    def _generate_dialogue(self, prompts, prompt_prefix=None):
        return self.interact(prompts, prompt_prefix=prompt_prefix)