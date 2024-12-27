import os
import re
import json
import copy
import time
import torch
import random
import argparse
import warnings
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import concurrent.futures

import torch
from accelerate import PartialState
from diffusers import (
    StableDiffusionXLPipeline, 
    StableDiffusionPipeline, 
    UNet2DConditionModel, 
    EulerDiscreteScheduler,
    PixArtAlphaPipeline,
    DiffusionPipeline,
    DDIMScheduler
)
from diffusers.utils import load_image
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from accelerate.utils import gather_object
from photomaker import PhotoMakerStableDiffusionXLPipeline

# Suppress warnings
warnings.filterwarnings("ignore")

MODULE_MAPPER = {
    't2i': 'sdxl-lightning',
    'p-t2i': 'photomaker',
    'web': 'bing',
    'retrieval': 'image_db'
}

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_sdxl_model(cache_dir: str, device: str):
    """Load the SDXL Lightning diffusion model."""
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_8step_unet.safetensors"
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base, unet=unet, torch_dtype=torch.float16, cache_dir=cache_dir, variant="fp16"
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    return pipe.to(device)


cache_dir = './pretrained_diffusion_model'

parser = argparse.ArgumentParser()
parser.add_argument('--start-idx', type=int)
parser.add_argument('--end-idx', type=int)
args = parser.parse_args()

model = load_sdxl_model(cache_dir, 'cuda:2')

@torch.inference_mode()
def generate_image(target_instance, SAVE_PATH):

    target_image_uuid = target_instance['image_uuid']

    images = model(
        target_instance['image_description'],
        num_inference_steps=8, 
        guidance_scale=0,
        num_images_per_prompt=1
    ).images

    save_paths = []
    for idx, image in enumerate(images):
        save_paths.append(os.path.join(SAVE_PATH, f'{idx}:{target_image_uuid}.png'))
        image.save(os.path.join(SAVE_PATH, f'{idx}:{target_image_uuid}.png'))

    cp_instance = copy.deepcopy(target_instance)
    cp_instance['image_save_paths'] = save_paths
    return cp_instance


def batch_images(dataset, SAVE_PATH):

    target_dataset, non_target_dataset = [], []
    for instance in tqdm(dataset, total=len(dataset)):
        module = instance['image_alignment_module']
        model_id = MODULE_MAPPER[module]
        
        if model_id == 'sdxl-lightning':
            if os.path.exists(os.path.join(SAVE_PATH, '0:{}.png'.format(instance['image_uuid']))):
                non_target_dataset.append(instance)
                continue
            target_dataset.append(instance)
        else:
            non_target_dataset.append(instance)
    
    print('# of total dataset:', len(dataset))
    print('# of target dataset:', len(target_dataset))
    print('# of non-target dataset:', len(non_target_dataset))

    completions_per_process = []
    for batch in tqdm(target_dataset, total=len(target_dataset)):

        result = generate_image(batch, SAVE_PATH)
        completions_per_process.append(result)
    
    print('# of final dataset:', len(completions_per_process) + len(non_target_dataset))
    return completions_per_process + non_target_dataset

if __name__ == '__main__':

    for persona_seed_num in range(args.start_idx, args.end_idx):
        dataset = load_json(f'curated_stark/planner-parsed-openai/stark_{persona_seed_num}.json')
        SAVE_PATH = f'generated_image/plan-and-execute/sdxl/stark_{persona_seed_num}'
        os.makedirs(SAVE_PATH, exist_ok=True)
        
        generations = batch_images(dataset, SAVE_PATH)

        data_save_path = 'curated_stark/plan-and-execute/sdxl'
        os.makedirs(data_save_path, exist_ok=True)

        with open(os.path.join(data_save_path, f'stark_{persona_seed_num}.json'), 'w', encoding='utf-8') as f:
            json.dump(generations, f, ensure_ascii=False, indent='\t')
