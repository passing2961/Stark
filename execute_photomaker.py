import os
import gc
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

distributed_state = PartialState()

def memory_optimization():
    # memory deallocation
    gc.collect()

    # removing cache
    torch.cuda.empty_cache()

def load_photomaker_model(cache_dir: str):
    base_model_path = 'SG161222/RealVisXL_V3.0'
        
    photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", cache_dir=cache_dir, filename="photomaker-v1.bin", repo_type="model")
    
    pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="fp16",
    ).to(distributed_state.device)

    pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_ckpt),
        subfolder="",
        weight_name=os.path.basename(photomaker_ckpt),
        trigger_word="img"
    )
    pipe.id_encoder.to(distributed_state.device) #device)
    
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.fuse_lora()

    return pipe

cache_dir = './pretrained_diffusion_model'

parser = argparse.ArgumentParser()
parser.add_argument('--start-idx', type=int)
parser.add_argument('--end-idx', type=int)
args = parser.parse_args()

model = load_photomaker_model(cache_dir) 

negative_prompt = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
generator = torch.Generator(device=distributed_state.device).manual_seed(42)

@torch.inference_mode()
def generate_image(target_instance, SAVE_PATH):
 
    target_image_uuid = target_instance['image_uuid']

    
    num_steps = 50
    style_strength_ratio = 20
    start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
    if start_merge_step > 30:
        start_merge_step = 30
    
    input_id_path = target_instance['face_image_path']
    input_id_images = [load_image(input_id_path)]
    images = model(
        prompt=target_instance['modified_image_description'],
        input_id_images=input_id_images,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=num_steps,
        start_merge_step=start_merge_step,
        generator=generator,
        #guidance_scale=5,
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
        
        if model_id == 'photomaker':
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
    with distributed_state.split_between_processes(target_dataset) as batched_prompts:
        for batch in tqdm(batched_prompts, total=len(batched_prompts)):
            
            memory_optimization()
            result = generate_image(batch, SAVE_PATH)
            completions_per_process.append(result)

    completions_gather = gather_object(completions_per_process)
    completions = completions_gather[: len(target_dataset)]

    memory_optimization()

    print('# of final dataset:', len(completions) + len(non_target_dataset))
    return completions + non_target_dataset

if __name__ == '__main__':

    for persona_seed_num in range(args.start_idx, args.end_idx):
        dataset = load_json(f'curated_stark/planner-parsed-openai/stark_{persona_seed_num}.json')
        SAVE_PATH = f'generated_image/plan-and-execute/generator/stark_{persona_seed_num}'
        os.makedirs(SAVE_PATH, exist_ok=True)
        
        generations = batch_images(dataset, SAVE_PATH)

        data_save_path = 'curated_stark/plan-and-execute/generator'
        os.makedirs(data_save_path, exist_ok=True)

        with open(os.path.join(data_save_path, f'stark_{persona_seed_num}.json'), 'w', encoding='utf-8') as f:
            json.dump(generations, f, ensure_ascii=False, indent='\t')
