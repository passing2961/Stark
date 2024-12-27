# 이 코드는 최종 데이터셋 형태를 만들기 위함이다
# Structure
# unique_id
# commonsense


import os
import copy
import json
import pandas as pd
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import Levenshtein


from utils.etc_utils import load_jsonl, load_json


BASE_DIR = 'outputs/stark_v1:gpt-3.5-turbo-0125/dialogue'


error_f = open('wrong_mapping_list.txt', 'w', encoding='utf-8')


def is_directory_empty(directory_path):
    try:
        if not os.listdir(directory_path):
            return True
        else:
            return False
    except FileNotFoundError:
        print(f"Error: The directory '{directory_path}' does not exist.")
        return None

def detect_and_fix_utf8_errors(names):
    fixed_names = []
    for name in names:
        try:
            name.encode('utf-8')
            fixed_names.append(name)
        except UnicodeEncodeError:
            # Replace invalid characters with '?'
            fixed_name = name.encode('utf-8', 'ignore').decode('utf-8')
            fixed_names.append(fixed_name)
    return fixed_names

def load_dataset(target_persona_seed_num):
    dataset = []
    count = 0
    for subdir in tqdm(glob(os.path.join(BASE_DIR, 'persona_seed:*'))):
        persona_seed_num = int(subdir.split('persona_seed:')[-1])

        if persona_seed_num != target_persona_seed_num:
            continue

        for session_dir in glob(os.path.join(subdir, 'session_num:*')):
            if is_directory_empty(session_dir):
                continue
            
            path = os.path.join(session_dir, 'final_output.jsonl')
            try:
                result = load_jsonl(path)
            except FileNotFoundError as e:
                continue

            for instance in result:
                dialog_id = '{}:{}-{}'.format(persona_seed_num, instance['id'], instance['commonsense_relation'])
                #all_dataset[dialog_id].append(instance)
                cp_instance = copy.deepcopy(instance)
                cp_instance['unique_id'] = dialog_id
                dataset.append(cp_instance)

    remove_column = [
        'persona-attr_system_message', 'persona-attr_prompt', 'birthplace_alpha2_code', 'residence_alpha2_code',
        'persona-attr_generation', 'commonsense_prompt',
        'commonsense_system_message', 'commonsense_generation', 
        'narrative_prompt', 'event-graph_prompt', 'event-graph_generation',
        'mobile-device_prompt', 
        'dialogue_prompt', 'dialogue_generation', 
        'persona-attr:prompt_tokens', 'persona-attr:completion_tokens',
        'commonsense:prompt_tokens', 'commonsense:completion_tokens',
        'narrative:prompt_tokens', 'narrative:completion_tokens', 
        'event-graph:prompt_tokens', 'event-graph:completion_tokens',
        'mobile-device:prompt_tokens', 'mobile-device:completion_tokens',
        'dialogue:prompt_tokens', 'dialogue:completion_tokens'
    ]
    df = pd.DataFrame(dataset)
    df.drop(columns=remove_column, inplace=True)
    df = df.groupby('unique_id').apply(lambda x: x.sort_values('session_number')).reset_index(drop=True)
        
    return df

from openai import OpenAI
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def calculate_similarity(name1, name2):
    # Levenshtein 거리 계산
    distance = Levenshtein.distance(name1, name2)
    # 최대 길이의 문자열을 기준으로 유사도 계산
    max_len = max(len(name1), len(name2))
    similarity = 1 - distance / max_len
    return similarity

def call_api(prompt):
    completion = client.chat.completions.create(
        model='gpt-4',
        messages=[{"role": "system", "content": "You are a helpful assistant."}, 
        {"role": "user", "content": "{}".format(prompt)}],
        temperature=0.0,
        max_tokens=128,
        top_p=0.0,
        #stop='\n\n'
    )
    output = completion.choices[0].message.content.strip()
    return output


def process_utter(persona_seed_num, dialogue_unique_id, utter_id, speaker, utter, sharing_info, mobile_device):
    check_all_empty = all((isinstance(v, list) and len(v) == 0) or (v == '') for v in sharing_info.values())
    if check_all_empty:
        return {
            'utter_id': utter_id,
            'speaker': speaker,
            'utter': utter,
            'sharing_info': ''
        }
    
    only_rationale_exist = (
        all(v == '' for k, v in sharing_info.items() if k != 'rationale') and
        sharing_info.get('rationale', '') != ''
    )
    if only_rationale_exist:
        return {
            'utter_id': utter_id,
            'speaker': speaker,
            'utter': utter,
            'sharing_info': ''
        }

    mobile_w_id_but_no_desc = (
        sharing_info.get('image_description', '') == '' and 
        sharing_info.get('image_source', '') == 'mobile' and
        ('image_id_from_mobile' in sharing_info.keys()) and
        sharing_info.get('image_id_from_mobile', '') != 'new added image'
    )
    if mobile_w_id_but_no_desc:
        cp_sharing_info = copy.deepcopy(sharing_info)
        mobile_image_id = sharing_info['image_id_from_mobile']
        if isinstance(mobile_image_id, str):
            if mobile_image_id == '':
                return {
                    'utter_id': utter_id,
                    'speaker': speaker,
                    'utter': utter,
                    'sharing_info': cp_sharing_info
                }
            elif mobile_image_id == 'new image added':
                return {
                    'utter_id': utter_id,
                    'speaker': speaker,
                    'utter': utter,
                    'sharing_info': cp_sharing_info
                }
            if mobile_image_id != '':
                mobile_image_id = int(mobile_image_id)
        
        try:
            cp_sharing_info['image_description'] = mobile_device[mobile_image_id]
        except (IndexError, TypeError) as e:
            error_f.write(f'[Error type: {e}]\nPersona seed num: {persona_seed_num}\nDialogue ID: {dialogue_unique_id}\n\n')
            return {
                'utter_id': utter_id,
                'speaker': speaker,
                'utter': utter,
                'sharing_info': cp_sharing_info
            }

        return {
            'utter_id': utter_id,
            'speaker': speaker,
            'utter': utter,
            'sharing_info': cp_sharing_info
        }

    internet_w_no_id_desc = (
        sharing_info.get('image_description', '') == '' and 
        sharing_info.get('image_source', '') == 'internet' and
        len(sharing_info.get('keywords', '')) != 0
    )
    if internet_w_no_id_desc:
        keywords = sharing_info['keywords']
        internet_prompt = f'Keywords: {keywords}\n\nGiven the keywords, your job is to generate a relevant concise image description, starting with an image of or a photo of.\nImage Description: '
        generated_image_desc = call_api(internet_prompt)
        cp_sharing_info = copy.deepcopy(sharing_info)
        cp_sharing_info['image_description'] = generated_image_desc
        return {
            'utter_id': utter_id,
            'speaker': speaker,
            'utter': utter,
            'sharing_info': cp_sharing_info
        }
    mobile_new_added_image_but_no_desc = (
        sharing_info.get('image_description', '') == '' and 
        sharing_info.get('image_source', '') == 'mobile' and
        sharing_info.get('image_id_from_mobile', '') == 'new added image'
    )
    if mobile_new_added_image_but_no_desc:
        return {
            'utter_id': utter_id,
            'speaker': speaker,
            'utter': utter,
            'sharing_info': sharing_info
        }

    new_added_image = (
        sharing_info.get('image_description', '') != '' and
        sharing_info.get('image_source', '') == 'mobile' and
        sharing_info.get('image_id_from_mobile', '') == 'new added image'
    )
    if new_added_image:
        return {
            'utter_id': utter_id,
            'speaker': speaker,
            'utter': utter,
            'sharing_info': sharing_info,
            'new_added_image': sharing_info['image_description']
        }

    return {
        'utter_id': utter_id,
        'speaker': speaker,
        'utter': utter,
        'sharing_info': sharing_info,
    }

AI_NAMES = [
    "AI assistant", "AIassistant", "AI 어시스턴트", "AI", "assistant", "AI Assistant", "Assistant",
    "AI_assistant", "ai", "АI assistant", "АI Assistant", "AI_Assistant", "AI_ASSISTANT", "ASSISTANT", "АИ assistant",
    "ai_assistant", "ai assistant", "ai_assistant"
]


def infer_name(dialogue, original_name, target_utter_id):

    redialogue = []
    for ele in dialogue:
        utter_id = ele['utterance_id']
        utter = ele['utterance']
        speaker = ele['speaker']
        sharing_info = ele['sharing_info']

        if utter_id == target_utter_id:
            redialogue.append(f'<SPEAKER>: [Sharing Image]')
            break

        if len(sharing_info) == 0:
            redialogue.append(f'{speaker}: {utter}')
        else:
            #image_desc = sharing_info['image_description']
            redialogue.append(f'{speaker}: [Sharing Image]')

    redialogue = '\n'.join(redialogue)
    prompt = f'Dialogue:\n{redialogue}\n\nGiven the dialogue, your job is to infer the speaker name (i.e., <SPEAKER>) when the "speaker" field is empty. What is the most appropriate speaker name among "AI Assistant" and "User"?\nAnswer: '
    output = call_api(prompt)
    
    if output.lower() == 'user':
        return original_name
    elif output.lower() == 'ai assistant':
        return 'AI Assistant'
    elif output.lower() == original_name.lower():
        return original_name
    else:
        print(output.lower(), original_name)
        assert False
    

def correct_name(gen_name, original_name, dialogue, utter_id):
    if gen_name.lower() == 'user':
        return original_name
    elif gen_name == '사용자':
        return original_name
    elif gen_name == '스프재':
        return original_name
    elif gen_name in AI_NAMES:
        return "AI Assistant"
    elif gen_name == '':
        return infer_name(dialogue, original_name, utter_id)
    else:

        return original_name

def process_session_dialogue(persona_seed_num, dialogue_unique_id, dialogue, current_mobile_device, original_name):
    redialogue = []
    current_mobile_device = copy.deepcopy(current_mobile_device)
    dialogue = copy.deepcopy(dialogue)
 
    for_infer_dialogue = []
    for item in dialogue:
        utter_id = item['utterance_id']
        speaker = item['speaker']
        utter = item['utterance']
        sharing_info = item['sharing_info']

        if speaker != original_name:
            corrected_name = correct_name(speaker, original_name, dialogue, utter_id)
            speaker = corrected_name

        if len(sharing_info) != 0:
            if isinstance(sharing_info, dict):
                processed_utter = process_utter(persona_seed_num, dialogue_unique_id, utter_id, speaker, utter, sharing_info, current_mobile_device)
                if 'new_added_image' in processed_utter.keys():
                    current_mobile_device.append({
                        'image_id': len(current_mobile_device),
                        'image_description': processed_utter['new_added_image']})
                
                redialogue.append({
                    'utter_id': processed_utter['utter_id'],
                    'speaker': processed_utter['speaker'],
                    'utter': processed_utter['utter'],
                    'sharing_info': processed_utter['sharing_info']
                })
            elif isinstance(sharing_info, list):
                processed_sharing_info = []
                for sharing_item in sharing_info:
                    _p_utter = process_utter(persona_seed_num, dialogue_unique_id, utter_id, speaker, utter, sharing_item, current_mobile_device)

                    if 'new_added_image' in _p_utter.keys():
                        current_mobile_device.append({
                            'image_id': len(current_mobile_device),
                            'image_description': _p_utter['new_added_image']})

                    processed_sharing_info.append({
                        'rationale': _p_utter['sharing_info']['rationale'],
                        'image_description': _p_utter['sharing_info']['image_description'],
                        'image_source': _p_utter['sharing_info']['image_source'],
                        'keywords': _p_utter['sharing_info']['keywords']
                    })
                
                redialogue.append({
                    'utter_id': utter_id,
                    'speaker': speaker,
                    'utter': utter,
                    'sharing_info': processed_sharing_info
                })
            
        else:
            redialogue.append({
                'utter_id': utter_id,
                'speaker': speaker,
                'utter': utter,
                'sharing_info': ''
            })

    return redialogue, copy.deepcopy(current_mobile_device)

def dump_output(outputs, file_name=None):
    f = open(file_name, 'w', encoding='utf-8') 
    for output in outputs:
        f.write(json.dumps(output) + '\n')
    f.close()
    
def process_episode(df, dialogue_unique_id, persona_seed_num):
    """Epsiode processing"""

    session_num_list = df['session_number'].unique()
    episode = []
    for session_idx, session_num in enumerate(session_num_list):
        session_dict = df[df['session_number'] == session_num].to_dict(orient='list')

        t_keys = [
            'dialogue:history_event', 
            'dialogue:event', 'dialogue:date',
            #'dialogue:last_date', 
            'dialogue:time_interval',
            'dialogue:experience'
        ]
        tmp = dict()

        for t_k in t_keys:
            tmp[f'seesion{session_num}:{t_k}'] = str(session_dict[t_k][0])
        
        name = session_dict['name'][0]
        dialogue = session_dict['parsed_dialogue_generation'][0]
        if session_idx == 0:
            mobile_device = session_dict['dialogue:mobile_device'][0]
        assert session_num == session_dict['session_number'][0]
        
        tmp[f'session{session_num}:mobile_device'] = str(mobile_device)
        
        processed_session_dialogue, mobile_device = process_session_dialogue(persona_seed_num, dialogue_unique_id, dialogue, mobile_device, name)
        
        tmp[f'session{session_num}:dialogue'] = str(processed_session_dialogue)
        if session_idx == len(session_num_list) - 1:
            tmp[f'session{session_num}:last_added_mobile_device_image'] = str(mobile_device)

        episode.append(tmp)
    
    return episode

if __name__ == '__main__':

    for persona_seed_num in range(0, 1):
        dataset = load_dataset(persona_seed_num)

        unique_id_list = dataset['unique_id'].unique()
        
        all_dataset = []
        for uid in tqdm(unique_id_list, total=len(unique_id_list)):
            df = dataset[dataset['unique_id'] == uid]
            
            t_keys = [
                'name', 'age', 'gender', 'birthplace', 'residence',
                'persona_category', 'persona-attr:sent', 'persona-attr:key', 'persona-attr:value',
                'commonsense_relation', 'narrative_sentence_form', 'parsed_event-graph_generation',
            ]
            temp_data = dict()
            temp_data['unique_id'] = uid
            for t_k in t_keys:
                if t_k == 'parsed_event-graph_generation':
                    temp_data['event-sequence'] = str(df[t_k].apply(str).unique()[0])
                    
                else:
                    temp_data[t_k] = str(df[t_k].unique()[0])

            processed_episode = process_episode(df, uid, persona_seed_num)

            temp_data['episode'] = processed_episode
            all_dataset.append(temp_data)
            

        base_save_dir = 'Stark'
        os.makedirs(base_save_dir, exist_ok=True)
        with open(os.path.join(base_save_dir, f'stark_{persona_seed_num}.json'), 'w', encoding='utf-8') as f:
           json.dump(all_dataset, f, ensure_ascii=False, indent='\t')

    error_f.close()