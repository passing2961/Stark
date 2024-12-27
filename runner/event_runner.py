import os
import copy
import random
from tqdm import tqdm
from collections import defaultdict
from glob import glob

from .base_runner import BaseRunner, console
from utils.etc_utils import load_jsonl, load_txt


SYSTEM_MESSAGE = """You should generate a temporal event graph composed of daily events occuring in a person's life. The temporal event graph contains nodes and edges. Each node represents a daily event which is written in two or three sentences. Each edge represents the casual relationship between two nodes (events), i.e., a past event -> current event. The current event is determined by how much time has passed since the past event and what personal experiences were had during that period. You must generate the temporal event graph following the guidelines below.

[Guideline]
- The graph is represented in the form of a json list.
- Each entry is a python dictionary containing the following keys: "id", "event", "date", "caused_by".
- The "id" field contains a unique identifier for the current event.
- The "event" field contains a description of the current event.
- The "date" field contains a specific date of the current event and is represented in the form of "%Y.%m.%d".
- The "caused_by" field represents the edge (i.e., a past event) and is represented in the form of a python dictionary containing the following keys: "caused_by:id", "caused_by:time_interval", "caused_by:experience_op", "caused_by:experience".
- The "caused_by:id" field contains an "id" of the past event that has caused the current event.
- The "caused_by:time_interval" field contains a time interval between the past event and the current event.
- The "caused_by:experience_op" field contains an episodic experience operation.
- The "caused_by:experience" field contains a short description of the added or updated episodic experience.
- The unit of time interval is ["hour", "day", "week", "month", "year"].
- The selected time interval should be formatted as "<base number> <time interval unit>".
- List of the episodic experience operation is ["add", "update"].
- The "add" operation refers to an operation that adds a new experience that have not been encountered in the past.
- The "update" operation refers to an operation that updates a past experience with a new experience.
- Events/Experiences can be positive or negative events or experiences.
- Events in the "caused_by:id" field should occur on dates before the current event that they have caused.
- If there is no entry of "caused_by" field, then you should generate an empty dictionary.
- Each event must be written in the present tense.
- The year in the "date" field must be until April 2024. 
- You should generate the temporal event graph based on commonsense or a world model."""


class EventRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)
        
        self.save_dir = os.path.join(self.output_base_dir, 'event-graph', f'persona_seed:{args.persona_seed_num}')
        os.makedirs(self.save_dir, exist_ok=True)
        
        self._load_prompt_template()
        self.last_save_chunk_idx_file = os.path.join(self.save_dir, 'last_save_chunk_idx.txt')
    
    @property
    def system_msg(self):
        return SYSTEM_MESSAGE

    @property
    def prompt_prefix(self):
        return "event-graph"

    def _load_prompt_template(self):
        self.event_template = load_txt('./templates/multi_modal_event.txt') 

    def _load_narrative(self):
        return load_jsonl(os.path.join(self.output_base_dir, 'album', 'final_output.jsonl'))
    
    def prepare_prompt(self):
        narrative = load_jsonl(os.path.join(self.output_base_dir, 'narrative', f'persona_seed:{self.args.persona_seed_num}', 'final_output.jsonl'))
        console.log('[{}] # of Total personal narrative: {}'.format(self.__class__.__name__, len(narrative)))

        if self.args.debug:
            
            narrative = random.sample(narrative, self.args.debug_sample_num)
        
        prompts = []
        for instance in tqdm(narrative, total=len(narrative)):
            
            _prompt = self.event_template.format(
                event=instance['narrative_generation'],
                name=instance['name'],
            )

            cp_instance = copy.deepcopy(instance)
            cp_instance[f'{self.prompt_prefix}_prompt'] = _prompt

            prompts.append(cp_instance)
        
        return prompts

    def check_json_key_exist(self, parsed_event):
        for instance in parsed_event:
            
            for key in ['id', 'event', 'date', 'caused_by']:
                if key not in instance.keys():
                    return False

            if instance['id'] != 1:
                for key in ['id', 'time_interval', 'experience_op', 'experience']:
                    if f'caused_by:{key}' not in instance['caused_by'].keys():
                        return False

        return True

    def parse_and_filter(self, generations):
        results, fail_results = [], []
        stat = defaultdict(int)
        error = []
        
        for generation in tqdm(generations, total=len(generations)):
            try:
                parsed_event = eval(self._parse_json_list(generation[f'{self.prompt_prefix}_generation']))
            except (SyntaxError, TypeError, NameError) as e:
                console.log(self._parse_json_list(generation[f'{self.prompt_prefix}_generation']), style='warning')
                stat[f'{self.prompt_prefix}_syntax:fail'] += 1
                error.append(str(e))
                continue
 
            if not self.check_json_key_exist(parsed_event):
                stat[f'{self.prompt_prefix}_key_existence:fail'] += 1
                fail_results.append(parsed_event)
                continue
            
            #replaced_event = self._replace_wrong_key(parsed_event)

            cp_generation = copy.deepcopy(generation)
            cp_generation[f'parsed_{self.prompt_prefix}_generation'] = parsed_event
            results.append(cp_generation)

        stat['event:total'] = len(generations)

        self.dump_output(results, os.path.join(self.save_dir, 'final_output.jsonl'))
        self.dump_report(stat, os.path.join(self.save_dir, 'report.txt'))
        self.dump_error_message(error, os.path.join(self.save_dir, 'error_message.json'))
        self.dump_json_output(fail_results, os.path.join(self.save_dir, 'fail_output.json'))

    def _replace_caused_by_key(self, instance, target_key=None):
        if target_key in instance['caused_by']:
            postfix = target_key.split('caused_by_')[-1]
            instance['caused_by'][f'caused_by:{postfix}'] = instance['caused_by'].pop(target_key)
        
        return instance

    def _replace_wrong_key(self, parsed_event):
        replaced_event = []
        for instance in parsed_event:
            
            if 'caused_by:' in instance:
                instance['caused_by'] = instance['caused_by:']

            if 'caused_by_id' in instance['caused_by']:
                for target_key in ['caused_by_id', 'caused_by_time_interval', 'caused_by_experience_op', 'caused_by_experience']:
                    instance = self._replace_caused_by_key(instance, target_key)
            
            replaced_event.append({
                'id': instance['id'],
                'event': instance['event'],
                'date': instance['date'],
                'caused_by': instance['caused_by'],
                #'mobile_device':instance['mobile_device']
            })
        return replaced_event

    def _generate_event_graph(self, prompts, prompt_prefix=None):
        return self.interact(prompts, prompt_prefix=prompt_prefix)