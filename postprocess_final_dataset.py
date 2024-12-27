import os
import json
import copy
import random
from tqdm import tqdm


def load_json(datadir):
    with open(datadir, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_jsonl(datadir: str):
    output = []
    with open(datadir, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            output.append(json.loads(line))
    return output

def dump_json_output(outputs, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent='\t')

def dump_jsonl_output(outputs, file_name=None):
    f = open(file_name, 'w', encoding='utf-8') 
    for output in outputs:
        f.write(json.dumps(output) + '\n')
    f.close()


def load_dataset():
    all_stark = []
    for persona_seed_num in range(0, 1):
        stark = load_json(os.path.join(f'./Stark/stark_{persona_seed_num}.json'))
        
        all_stark.extend(stark)
    
    return all_stark

def process_dialog(dialog):
    dialogue = eval(dialog)
    cnt = 0
    new_cnt = 0
    redialog = []
    for item in dialogue:
        utter_id = item['utter_id']
        speaker = item['speaker']
        utter = item['utter']
        sharing_info = item['sharing_info']

        cp_item = copy.deepcopy(item)
        if utter == '' and len(sharing_info) == 0:
            cp_item['utter'] = '<non verbal>'
        
        redialog.append(cp_item)
    
    return redialog

def process_dataset(dataset):

    re_dataset = []
    for instance in dataset:
        uuid = instance['unique_id']
        name = instance['name']

        episode = instance['episode']

        cp_instance = copy.deepcopy(instance)

        re_epi = []
        for idx, session in enumerate(episode):
            session_dialog = session[f'session{idx+1}:dialogue']
            p_dialog = process_dialog(session_dialog)
            
            cp_session = copy.deepcopy(session)
            cp_session[f'session{idx+1}:dialogue'] = p_dialog
            re_epi.append(cp_session)
        
        cp_instance['episode'] = re_epi
        re_dataset.append(cp_instance)
    
    return re_dataset
    #return all_count

if __name__ == '__main__':

    dataset = load_dataset()
    print(len(dataset))
    save_dir = 'Stark/post-process'
    os.makedirs(save_dir, exist_ok=True)

    processed_dataset = process_dataset(dataset)
    print(len(processed_dataset))
    dump_json_output(processed_dataset, os.path.join(save_dir, 'stark_0.json'))