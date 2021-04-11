from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import PIL
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import get_image_backend

from util import get_train_file_path, get_test_file_path, string_to_ints

class InputDatasetTest(Dataset):
    def __init__(self, df, transform, stoi, is_train=True):
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

        #numericalize the caption text
        inchi_str = self.inchis[idx]
        inchi_vec = string_to_ints(inchi_str, self.stoi)

        return sample, torch.tensor(inchi_vec), idx


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