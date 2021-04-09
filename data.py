from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import get_image_backend

from util import get_train_file_path

class InputDatasetTest(Dataset):
    def __init__(self, train, transform):
        self.img_paths=get_train_file_path(train['image_id']) # TODO
        self.inchis = train['InChI']
        self.loader=default_loader
        self.transform=transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        sample = self.loader(self.img_paths[idx])
        sample = self.transform(sample)
        inchi = self.inchis[idx]
        return sample, idx


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