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

import clip
import faiss
import yaml
import uuid
import requests
import concurrent.futures

# Suppress warnings
warnings.filterwarnings("ignore")



IMAGE_DB_SOURCE_NAMES = ["cc12m", "redcaps12m", "mathvision", "chartqa", "ai2d"] #, "gqa", "ocr_vqa", "textvqa", "vg"]
IMAGE_SEARCH_NUM = 5

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

class ImageDBLoader:
    """
    Load image list and image indices.
    """
    @staticmethod
    def load_image_db():
        image_list = dict()
        image_indices = dict()
        for image_dataset in IMAGE_DB_SOURCE_NAMES:
            data_dir = Path(f'../Sonny-PM/prepare_image_db/embeddings_folder/{image_dataset}/metadata')
            df = pd.concat(
                pd.read_parquet(parquet_file)
                for parquet_file in data_dir.glob('*.parquet')
            )

            ind = faiss.read_index(f'../Sonny-PM/prepare_image_db/index_folder/{image_dataset}/knn.index')
            image_list[image_dataset] = df['image_path'].tolist()
            image_indices[image_dataset] = ind

        return {
            'image_list': image_list,
            'image_indices': image_indices
        }

image_db = ImageDBLoader.load_image_db()
print(f'Load Image DB Done!')
image_mapper = {
    'cc12m': load_json('../Sonny-PM/prepare_image_db/image_mapper/cc12m.json'),
    'redcaps12m': load_json('../Sonny-PM/prepare_image_db/image_mapper/redcaps12m.json')
}
print(f'Load image mapper Done!')

MODULE_MAPPER = {
    't2i': 'sdxl-lightning',
    'p-t2i': 'photomaker',
    'web': 'bing',
    'retrieval': 'image_db'
}

def load_openai_clip_model(device: str):
    model, _ = clip.load('ViT-L/14@336px', device=device, jit=False)
    return model

model = load_openai_clip_model('cuda:0')

@torch.no_grad
def retrieve_image(instance, persona_seed_num):
    image_desc = instance['image_description']
    image_uuid = '{}:{}'.format(persona_seed_num, instance['image_uuid'])
    
    desc_tokens = clip.tokenize(image_desc, truncate=True)
    desc_feats = model.encode_text(desc_tokens.to('cuda:0'))
    desc_feats /= desc_feats.norm(dim=-1, keepdim=True)
    desc_embeds = desc_feats.cpu().detach().numpy().astype('float32')
    #print("Done get embedding")

    image_search_result = dict()
    for src_name in IMAGE_DB_SOURCE_NAMES:
        D, I = image_db['image_indices'][src_name].search(desc_embeds, IMAGE_SEARCH_NUM)

        tmp_result = []
        for item_D, item_I in zip(D[0], I[0]):
            if src_name in ['redcaps12m', 'cc12m']:
                target_mapper = image_mapper[src_name]

                tmp_result.append({
                    'image_path_from_db': image_db['image_list'][src_name][item_I], #target_mapper[image_db['image_list'][src_name][item_I]],
                    'clip_score': str(item_D)
                })
            else:
                tmp_result.append({
                    'image_path_from_db': image_db['image_list'][src_name][item_I],
                    'clip_score': str(item_D)
                })
        image_search_result[src_name] = tmp_result

    return image_search_result

def batch_images(dataset, persona_seed_num):

    target_dataset, non_target_dataset = [], []
    for instance in tqdm(dataset, total=len(dataset)):
        module = instance['image_alignment_module']
        model_id = MODULE_MAPPER[module]
        if model_id == 'image_db':
            
            target_dataset.append(instance)
        else:
            non_target_dataset.append(instance)

    final_dataset = []
    for target_instance in tqdm(target_dataset, total=len(target_dataset)):
        retrieved_results = retrieve_image(target_instance, persona_seed_num)
        cp_instance = copy.deepcopy(target_instance)
        cp_instance['db_searched_results'] = retrieved_results
        final_dataset.append(cp_instance)

    return final_dataset + non_target_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-idx', type=int)
    parser.add_argument('--end-idx', type=int)
    args = parser.parse_args()

    for persona_seed_num in range(args.start_idx, args.end_idx):

        dataset = load_json(f'curated_stark/planner-parsed-openai/stark_{persona_seed_num}.json')
        generations = batch_images(dataset, persona_seed_num)

        data_save_path = 'curated_stark/plan-and-execute/image_db'
        os.makedirs(data_save_path, exist_ok=True)

        with open(os.path.join(data_save_path, f'stark_{persona_seed_num}.json'), 'w', encoding='utf-8') as f:
            json.dump(generations, f, ensure_ascii=False, indent='\t')

