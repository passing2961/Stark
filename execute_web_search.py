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

import uuid
import requests
import concurrent.futures
from icrawler.builtin import BingImageCrawler


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

def search_image(target_instance, SAVE_PATH):
    image_uuid = target_instance['image_uuid']
    bing_crawler = BingImageCrawler(downloader_threads=8, storage={"root_dir": os.path.join(SAVE_PATH, image_uuid)})

    image_desc = target_instance['image_description']
    bing_crawler.crawl(keyword=image_desc, offset=0, max_num=10, filters=None)

    cp_instance = copy.deepcopy(target_instance)
    cp_instance['image_save_path'] = os.path.join(SAVE_PATH, image_uuid)
    return cp_instance

def batch_images(dataset, SAVE_PATH):
    target_dataset, non_target_dataset = [], []
    for instance in tqdm(dataset, total=len(dataset)):
        module = instance['image_alignment_module']
        model_id = MODULE_MAPPER[module]

        image_uuid = instance['image_uuid']

        
        if model_id == 'bing':
            if os.path.exists(os.path.join(SAVE_PATH, image_uuid)):
                non_target_dataset.append(instance)
                continue
            target_dataset.append(instance)
        else:
            non_target_dataset.append(instance)

    print('# of total dataset:', len(dataset))
    print('# of target dataset:', len(target_dataset))
    print('# of non-target dataset:', len(non_target_dataset))

    final_dataset = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = []

        for instance in tqdm(target_dataset, total=len(target_dataset)):
            cp_instance = copy.deepcopy(instance)

            future = executor.submit(search_image, cp_instance, SAVE_PATH)
            futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            ret = future.result()
            final_dataset.append(ret)
    
    return final_dataset + non_target_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-idx', type=int)
    parser.add_argument('--end-idx', type=int)
    args = parser.parse_args()

    for persona_seed_num in range(args.start_idx, args.end_idx):

        dataset = load_json(f'curated_stark/planner-parsed-openai/stark_{persona_seed_num}.json')
        SAVE_PATH = f'generated_image/plan-and-execute/web_searcher/stark_{persona_seed_num}'
        os.makedirs(SAVE_PATH, exist_ok=True)
        
        generations = batch_images(dataset, SAVE_PATH)

        data_save_path = 'curated_stark/plan-and-execute/web_searcher'
        os.makedirs(data_save_path, exist_ok=True)

        with open(os.path.join(data_save_path, f'stark_{persona_seed_num}.json'), 'w', encoding='utf-8') as f:
            json.dump(generations, f, ensure_ascii=False, indent='\t')
