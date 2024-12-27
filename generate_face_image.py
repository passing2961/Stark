import os
import json
import copy
import random
import argparse
from tqdm import tqdm

import torch
from diffusers import (
    StableDiffusionXLPipeline, 
    StableDiffusionPipeline, 
    UNet2DConditionModel, 
    EulerDiscreteScheduler,
    PixArtAlphaPipeline,
    DiffusionPipeline,
    DDIMScheduler
)
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download


IMAGE_DB_SOURCE_NAMES = ["coco_train2017", "gqa", "ocr_vqa", "textvqa", "vg"]
IMAGE_SEARCH_NUM = 5

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

def make_human_attribute_sentence(human_attr, age, gender, birthplace):
    style_type = random.choice(STYLE_TYPES)
    gender = gender.lower()
    template_parts = [f'A {style_type} shot, a {age}-years-old {gender} from {birthplace},']
    
    excluded_categories = {'face', 'overall-style'}
    suffix_map = {
        'hair': ' hair',
        'belts': ' belt',
        'scarf': ' scarf',
        'headband': ' headband',
        'headscarf': ' headscarf',
        'veil': ' veil',
        'socks': ' socks',
        'ties': ' tie'
    }
    
    for category, attribute in human_attr.items():
        if category in excluded_categories:
            continue
        
        if category == 'hair' and 'wears' in attribute:
            continue
        
        order_keys = HUMAN_ATTR_KEY_ORDER.get(category, [])
        for order_key in order_keys:
            attr_value = attribute.get(order_key)
            if attr_value:
                template_parts.append(f' {attr_value}')
        
        suffix = suffix_map.get(category)
        if suffix:
            template_parts[-1] += suffix
        
        template_parts[-1] += ','

    template = ''.join(template_parts).rstrip(',') + '.'
    
    return template

def load_human_attribute_pool():
    return load_jsonl(os.path.join('./datasets/cosmic/human_attribute_pool.jsonl'))

def load_sdxl_diffusion_model(cache_dir: str, device: str):
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_8step_unet.safetensors"

    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base, unet=unet, torch_dtype=torch.float16, cache_dir=cache_dir, variant="fp16"
    ).to(device)

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    return pipe

def process_face(stark, human_attribute_pool, pipeline, persona_seed_num, save_dir):
    total_id_keys = list(set([ele['unique_id'] for ele in stark]))
    total_id_num = len(total_id_keys)
    sampled_human_attribute = random.sample(human_attribute_pool, total_id_num)
    id2human_attr = {key: sampled_human_attribute[i] for i, key in enumerate(total_id_keys)}

    final_results = []
    for idx, instance in enumerate(tqdm(stark, total=len(stark))):
        uuid = instance['unique_id']
        age = instance['age']
        gender = instance['gender']
        birthplace = instance['birthplace']

        human_attr = id2human_attr[uuid]
        human_attr_sent = make_human_attribute_sentence(human_attr, age, gender, birthplace)

        face_image = pipeline(human_attr_sent, num_inference_steps=8, guidance_scale=0).images[0]
        face_image.save(os.path.join(save_dir, f'{uuid}.png'))
        
        cp_instance = copy.deepcopy(instance)
        cp_instance['face_description'] = human_attr_sent
        cp_instance['face_image_path'] = os.path.join(save_dir, f'{uuid}.png')
        final_results.append(cp_instance)
    
    return final_results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--start-idx', type=int)
    parser.add_argument('--end-idx', type=int)
    parser.add_argument('--device', type=str)
    args = parser.parse_args()

    human_attribute_pool = load_human_attribute_pool()
    pipeline = load_sdxl_diffusion_model('./pretrained_diffusion_model', args.device)
    
    for persona_seed_num in range(args.start_idx, args.end_idx):
        stark = load_json(os.path.join(f'./Stark/post-process/stark_{persona_seed_num}.json'))
        
        save_dir = f'generated_image/human-face/stark_{persona_seed_num}'
        os.makedirs(save_dir, exist_ok=True)

        processed_results = process_face(stark, human_attribute_pool, pipeline, persona_seed_num, save_dir)
        
        curated_save_dir = f'curated_stark/human-face'
        os.makedirs(curated_save_dir, exist_ok=True)

        dump_json_output(processed_results, os.path.join(curated_save_dir, f'stark_{persona_seed_num}.json'))