import os
import re
import json
import copy
import time
import random
import argparse
from tqdm import tqdm

import uuid
import openai
from openai import OpenAI
import concurrent.futures


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

SYSTEM_MESSAGE = """Your job is to determine the most appropriate module from a list of models to process the input request. Please select one module from the following list:

Personalized Text-to-Image Generator: This module generates personalized images from a given description and a human face image. For example, if you provide a face image and a description like “A selfie of Tom smiling at the Golden State Warriors' arena during a game,” the module will generate a customized realistic human image. Note that when you generate the answer, you must generate the module name and modified image description together. The modified image description MUST include a strict format: “<class_word> [img]”. <class_word> represents the identity of a human, such as a man, woman, girl, boy, or young boy, etc. [img] denotes the special token. You must not omit this strict format, and you must keep the original image description as it is and only add this strict format.

Web Search: This module finds related images from the internet in real-time based on the given user's input image description. The image description is primarily related to the latest information. Therefore, this method is useful when up-to-date information is needed.

Image Database Retrieval: This module finds relevant images from a pre-built image database based on the given user's input image description. To build an image database containing images on various topics, images are collected from the RedCaps, Conceptual Captions 12M (CC12M), ChartQA, AI2D, and MathVision datasets. Descriptions related to each dataset are as follows:
- RedCaps: This is a large-scale dataset of 12M image-text pairs collected from Reddit. Images and captions from Reddit depict and describe a wide variety of objects and scenes.
- CC12M: This is a dataset with 12 million image-text pairs specifically meant to be used for vision and language pre-training. It is larger and covers a much more diverse set of visual concepts than the Conceptual Captions (CC3M).
- ChartQA: This is a large-scale ChartQA dataset with real-world charts and human-authored question-answer pairs. This dataset covers 9.6K chart images.
- AI2D: This is a dataset of over 5,000 grade school science diagrams with over 150,000 rich annotations, their ground truth syntactic parses, and more than 15,000 corresponding multiple choice questions.
- MathVision: This dataset is a meticulously curated collection of 3,040 high-quality mathematical problems with visual contexts sourced from real math competitions. Spanning 16 distinct mathematical disciplines and graded across 5 levels of difficulty.

For example,

Name: Tom
Gender: Male
Age: 21
Image Description: A selfie of Tom smiling at the Golden State Warriors' arena during a game
Module: Personalized Text-to-Image Generator
Modified Image Description: A selfie of a young man [img] smiling at the Golden State Warriors' arena during a game

Name: Tom
Gender: Male
Age: 21
Image Description: A screenshot of chatbot development code using Python
Module: Image Database Retrieval

Name: Tom
Gender: Male
Age: 21
Image Description: A photo of Manchester United lifting the 2023-24 FA Cup trophy
Module: Web Search"""

EXISTING_UUIDS = set()
IMAGE_MODULE_MAPPER = {
    'Text-to-Image Generator': 't2i',
    'Personalized Text-to-Image Generator': 'p-t2i',
    'Web Search': 'web',
    'Image Database Retrieval': 'retrieval'
}

with open('./templates/plan_template.txt', 'r') as f:
    prompt_template = f.read()

def generate_unique_uuid():
    while True:
        new_uuid = str(uuid.uuid4())
        if new_uuid not in EXISTING_UUIDS:
            EXISTING_UUIDS.add(new_uuid)
            return new_uuid

def dump_json_output(outputs, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent='\t')


def call_openai_api(prompt_input, prompt_prefix=None):
    prompt = prompt_input[f'{prompt_prefix}_prompt']

    while True:
        try:
            completion = client.chat.completions.create(
                model='gpt-3.5-turbo-0125',
                messages=[{"role": "system", "content": SYSTEM_MESSAGE}, 
                {"role": "user", "content": "{}".format(prompt)}],
                temperature=0.9,
                max_tokens=1024,
                top_p=0.95,
                frequency_penalty=1.0, 
                presence_penalty=0.6
                #stop='\n\n'
            )
            break
        except (RuntimeError, openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError) as e:
            print("Error: {}".format(e))
            time.sleep(2)
            continue
        
    output = completion.choices[0].message.content.strip()
    completion_tokens = completion.usage.completion_tokens
    prompt_tokens = completion.usage.prompt_tokens
    
    prompt_input[f'{prompt_prefix}_generation'] = output
    prompt_input[f'{prompt_prefix}:prompt_tokens'] = prompt_tokens
    prompt_input[f'{prompt_prefix}:completion_tokens'] = completion_tokens

    return prompt_input    

def interact(prompts, prompt_prefix=None):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = []

        for instance in tqdm(prompts, total=len(prompts)):
            cp_instance = copy.deepcopy(instance)

            future = executor.submit(call_openai_api, cp_instance, prompt_prefix=prompt_prefix)
            futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            ret = future.result()
            results.append(ret)

    return results

def load_json(datadir):
    with open(datadir, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_dialogue(session_dialogue):

    annotated_dialogue_image = []

    redialogue = []
    for item in session_dialogue:
        utter_id = item['utter_id']
        speaker = item['speaker']
        utter = item['utter']
        sharing_info = item['sharing_info']

        cp_item = copy.deepcopy(item)

        if len(sharing_info) != 0:
            if isinstance(sharing_info, list):
                re_sharing_info = []
                for ele in sharing_info:
                    image_desc = ele['image_description']
                    image_uuid = generate_unique_uuid()
                    cp_ele = copy.deepcopy(ele)
                    cp_ele['image_uuid'] = image_uuid
                    re_sharing_info.append(cp_ele)
                    annotated_dialogue_image.append({
                        'image_uuid': image_uuid,
                        'image_description': image_desc
                    })

                cp_item['sharing_info'] = re_sharing_info
                
            elif isinstance(sharing_info, dict):
                
                image_desc = sharing_info['image_description']
                cp_sharing_info = copy.deepcopy(sharing_info)

                if image_desc != '':
                    image_uuid = generate_unique_uuid()
                    cp_sharing_info['image_uuid'] = image_uuid
                    annotated_dialogue_image.append({
                        'image_uuid': image_uuid,
                        'image_description': image_desc
                    })
                    cp_item['sharing_info'] = cp_sharing_info

            redialogue.append(cp_item)
        else:
            redialogue.append(cp_item)

    return redialogue, annotated_dialogue_image

def process_dataset(dataset):

    final_results = []
    final_image_results = []
    for idx, instance in enumerate(tqdm(dataset, total=len(dataset))):
        name = instance['name']
        gender = instance['gender']
        age = instance['age']
        face_image_path = instance['face_image_path']
        face_image_desc = instance['face_description']

        episode = instance['episode']
        episode_len = len(episode)

        new_episode = []
        # first, collect all mobile device and annotate unique id
        session_mobile_device = eval(episode[-1][f'session{episode_len}:last_added_mobile_device_image'])
        annotated_mobile_device = []
        annotated_dict = dict()
        for item in session_mobile_device:
            mobile_uuid = generate_unique_uuid()
            annotated_mobile_device.append({
                'image_uuid': mobile_uuid,
                'image_id': item['image_id'],
                'image_description': item['image_description']
            })
            annotated_dict['{}:{}'.format(item['image_id'], item['image_description'])] = mobile_uuid
            final_image_results.append({
                'image_uuid': mobile_uuid,
                'image_description': item['image_description'],
                'name': name,
                'gender': gender,
                'age': age,
                'face_image_path': face_image_path,
                'face_description': face_image_desc
            })

        # second, collect all image from the dialogue
        for session_idx, session in enumerate(episode):
            
            session_dialogue = session[f'session{session_idx+1}:dialogue']
            session_mobile_device = eval(session[f'session{session_idx+1}:mobile_device'])
            
            re_session_dialogue, annotated_session_dialogue = process_dialogue(session_dialogue)

            for ele_anno_session in annotated_session_dialogue:
                final_image_results.append({
                    'image_uuid': ele_anno_session['image_uuid'],
                    'image_description': ele_anno_session['image_description'],
                    'name': name,
                    'gender': gender,
                    'age': age,
                    'face_image_path': face_image_path,
                    'face_description': face_image_desc
                })

            re_session_mobile_device = []
            for ele_s_m in session_mobile_device:
                re_session_mobile_device.append({
                    'image_id': ele_s_m['image_id'],
                    'image_uuid': annotated_dict['{}:{}'.format(ele_s_m['image_id'], ele_s_m['image_description'])],
                    'image_description': ele_s_m['image_description']
                })
            
            cp_session = copy.deepcopy(session)
            cp_session[f'session{session_idx}:dialogue'] = re_session_dialogue
            cp_session[f'session{session_idx}:mobile_device'] = re_session_mobile_device
            new_episode.append(cp_session)
        
        cp_instance = copy.deepcopy(instance)
        cp_instance['episode'] = new_episode
        final_results.append(cp_instance)
    
    print('# of final results:', len(final_results))
    print('# of final image results:', len(final_image_results))
    return final_results, final_image_results

def process_planner(dataset):

    re_dataset = []
    for instance in tqdm(dataset, total=len(dataset)):
        image_uuid = instance['image_uuid']
        image_desc = instance['image_description']

        prompt = prompt_template.format(
            name=instance['name'],
            age=instance['age'],
            gender=instance['gender'],
            image_description=image_desc
        )
        cp_instance = copy.deepcopy(instance)
        cp_instance['image_desc_prompt'] = prompt
        re_dataset.append(cp_instance)
    
    return re_dataset

def parse_mobile_device_using_regex(generation):
    # Define the regex pattern
    pattern = r"(?P<module>.*)\nModified Image Description:\s*(?P<description>.*)"

    # Use re.search to find the match
    match = re.search(pattern, generation, re.DOTALL)

    if match:
        module = match.group("module")
        description = match.group("description")
        return {
            'module': module,
            'description': description
        }
    else:
        
        if 'A photo of a woman [img] enjoying a serene landscape with a colorful sunset pinned on a Pinterest board dedicated to nature photography' in generation:
            return {
                'module': 'Personalized Text-to-Image Generator',
                'description': 'A photo of a woman [img] enjoying a serene landscape with a colorful sunset pinned on a Pinterest board dedicated to nature photography'
            }
        print(generation, match)
        raise ValueError("Wrong")


def process_img_tags(text):
    parts = text.split("[img]")
    processed_text = '[img]'.join(parts[:2])
    return processed_text + ' '.join([ele.strip() for ele in parts[2:]])

def parse_and_filter(generations):
    results = []
    for generation in tqdm(generations, total=len(generations)):
        parsed_generation = generation['image_desc_generation']

        parsed_generation = parsed_generation.replace('Module:', '').strip()
        
        if 'Personalized Text-to-Image Generator' in parsed_generation:
            parsed_result = parse_mobile_device_using_regex(parsed_generation)
            for module_name in ['Personalized Text-to-Image Generator', 'Image Database Retrieval', 'Web Search']:
                if module_name in parsed_result['module']:
                    module = module_name
                    break
            module = IMAGE_MODULE_MAPPER[module]
            image_desc = parsed_result['description']
            if image_desc.count('[img]') > 1:
                
                modified_desc = process_img_tags(image_desc)

            elif image_desc.count('[img]') == 0:
                modified_desc = image_desc + ' [img]'
            elif image_desc.count('[img]') == 1:
                modified_desc = image_desc

            modified_desc = modified_desc.replace('[img]', 'img')
                
        else:
            if 'Modified Image Description' in parsed_generation:
                parsed_generation = parsed_generation.split('Modified Image Description: ')[0].strip()
            
            for module_name in ['Personalized Text-to-Image Generator', 'Image Database Retrieval', 'Web Search']:
                if module_name in parsed_generation:
                    if module_name == 'Personalized Text-to-Image Generator':
                        assert False, f'{parsed_generation}'

                    parsed_generation = module_name
                    break
            
            try:
                module = IMAGE_MODULE_MAPPER[parsed_generation]
                modified_desc = ''
            except KeyError:
                module = 't2i'
                modified_desc = ''
        
        cp_generation = copy.deepcopy(generation)
        cp_generation['image_alignment_module'] = module
        cp_generation['modified_image_description'] = modified_desc
        results.append(cp_generation)
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-idx', type=int)
    parser.add_argument('--end-idx', type=int)
    parser.add_argument('--do-planner', action='store_true')
    args = parser.parse_args()

    if args.do_planner:
        for persona_seed_num in range(args.start_idx, args.end_idx):

            stark = load_json(f'curated_stark/planner-image-only/stark_{persona_seed_num}.json')

            processed_stark = process_planner(stark)
            generations = interact(processed_stark, prompt_prefix='image_desc')
            save_dir = f'curated_stark/planner-openai'
            os.makedirs(save_dir, exist_ok=True)
            dump_json_output(generations, os.path.join(save_dir, f'stark_{persona_seed_num}.json'))


            generations = load_json(f'curated_stark/planner-openai/stark_{persona_seed_num}.json')
            parsed_generations = parse_and_filter(generations)
            save_dir = f'curated_stark/planner-parsed-openai'
            os.makedirs(save_dir, exist_ok=True)
            dump_json_output(parsed_generations, os.path.join(save_dir, f'stark_{persona_seed_num}.json'))    
    
    else:
        for persona_seed_num in range(args.start_idx, args.end_idx):
            stark = load_json(f'curated_stark/human-face/stark_{persona_seed_num}.json')

            curated_stark, stark_images = process_dataset(stark)

            save_dir = f'curated_stark/planner-image-only'
            os.makedirs(save_dir, exist_ok=True)
            dump_json_output(stark_images, os.path.join(save_dir, f'stark_{persona_seed_num}.json'))

            curated_save_dir = f'curated_stark/planner-dialogue'
            os.makedirs(curated_save_dir, exist_ok=True)
            dump_json_output(curated_stark, os.path.join(curated_save_dir, f'stark_{persona_seed_num}.json'))