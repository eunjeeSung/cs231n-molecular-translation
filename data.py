from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import PIL
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import get_image_backend

from util import get_train_file_path, get_test_file_path, string_to_ints

class InputDatasetTest(Dataset):
    def __init__(self, df, transform, stoi, is_train=True, *args, **kwargs):
        self.img_paths = get_train_file_path(df['image_id']) if is_train \
                            else get_test_file_path(df['image_id'])
        self.inchis = df['InChI']
        self.loader = default_loader
        self.transform = transform
        self.stoi = stoi

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        sample = self.loader(self.img_paths[idx])
        sample = self.transform(sample)

        # (1) For whole InChI text
        #inchi_str = self.inchis[idx][9:]
        # inchi_vec = string_to_ints(inchi_str, self.stoi)
        
        # (2) For chemical note generator
        inchi_str = self.inchis[idx].split('/')[1]
        inchi_vec = string_to_ints(inchi_str, self.stoi)

        # (3) For m
        # for layerid, layer in enumerate(['m']):
        #     inchi_str = [ ''.join([splitlayer if splitlayer[0]==layer else "" for splitlayer in self.inchis[idx].split("/")])
        #                         for train_label in self.inchis[idx]]
        #     inchi_str = ['' if len(item)==0 else item[1:] for item in inchi_str][0]
        # mtoi = {'': 0, '0': 1, '0m1': 2, '1': 3, '1m0': 4}
        # inchi_vec = [mtoi[inchi_str]]

        return sample, torch.tensor(inchi_vec), torch.tensor(idx)


def pil_loader(path: str) -> Image.Image: #copied from torchvision
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def default_loader(path: str) -> Any:
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)