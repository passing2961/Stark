import os

from tqdm import tqdm
import pyarrow.parquet as pq
import pandas as pd
from joblib import Parallel, delayed
from joblib_progress import joblib_progress

from .base_runner import BaseRunner, console


def _rebuild_cc12m_dataset(filename):
    if '.jpg' not in filename:
        return None
    else:
        return filename
        
class AlignmentRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)

        self._load_image_file()

    def _flatten_dataset(self, dataset):
        return [
            ele for ele in dataset if ele is not None
        ]

    def system_msg(self):
        return "You are a helpful assistant."

    def prompt_prefix(self):
        return "persona-attr"
        
    def _load_image_file(self):
        cc12m_path = '/home/yjlee/workspace/ICCV2023/pipeline/data_collection/image_files/cc12m'

        subdir = os.listdir(cc12m_path)[:100]
        cc12m_filenames = []
        for _dir in tqdm(subdir, total=len(subdir)):
            if '.parquet' in _dir or '.json' in _dir:
                continue
            
            for ele in os.listdir(os.path.join(cc12m_path, _dir)):
                cc12m_filenames.append(os.path.join(cc12m_path, _dir, ele))
        
        console.log('[{}] # of CC12M dataset: {}'.format(self.__class__.__name__, len(cc12m_filenames)))

        with joblib_progress("Loading CC12M dataset...", total=len(cc12m_filenames)):
            cc12m_dataset = Parallel(n_jobs=32)(delayed(_rebuild_cc12m_dataset)(filename) for filename in cc12m_filenames)
        
        cc12m_filenames = self._flatten_dataset(cc12m_dataset)
        
        console.log('[{}] # of CC12M dataset: {}'.format(self.__class__.__name__, len(cc12m_filenames)))


    def run(self):
        return None