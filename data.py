from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import numpy as np

import PIL
from PIL import Image, ImageOps, ImageFilter

from skimage import data, feature, filters, morphology, util
from skimage.filters.rank import median
from skimage.morphology import disk, opening, closing

import torch
import torchvision.transforms.functional
from torch.utils.data import Dataset
from torchvision import get_image_backend

from util import get_train_file_path, get_test_file_path, string_to_ints, c_string_to_ints, t_string_to_ints

class InputDatasetTest(Dataset):
    def __init__(self, df, transform, dictionaries, is_train=True, *args, **kwargs):
        self.img_paths = get_train_file_path(df['image_id']) if is_train \
                            else get_test_file_path(df['image_id'])
        self.loader = default_loader
        self.transform = transform
        self.general_to_i, self.t_to_i, self.m_to_i, self.s_to_i = dictionaries
        self.df = df

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # image = self.loader(self.img_paths[idx])
        # w, h = image.size
        # if h > w:
        #     image = functional.rotate(image, 90)

        # image = ImageOps.invert(image)


        im = Image.open(self.img_paths[idx])
        w, h = im.size
        if h > w:
            im = functional.rotate(image, 90)        
        im1 = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
        im2 = im1.filter(ImageFilter.SHARPEN)
        selem = disk(1)
        im3 = opening(closing(filters.sobel(im2),selem),selem)
        final = median(im3,selem)
        # final = Image.fromarray(np.repeat(255*final, 3, axis=0).astype(np.uint8))
        final = Image.fromarray(final)
        final = final.convert('RGB')

        
        image = self.transform(final)

        # (1) Chemical note -> dict
        note_str = self.df['chemical_notation'][idx]
        note_vec = torch.tensor(string_to_ints(note_str, self.general_to_i))

        # (2) C layer: 1-9(2)8-15-... -> remove -, dict
        c_str = self.df['c'][idx]
        c_vec = torch.tensor(c_string_to_ints(c_str, self.general_to_i))

        # (3) H layer: 4,6-7H,1-3,5H2,(H,18,22)... -> dict
        h_str = self.df['h'][idx]
        h_vec = torch.tensor(string_to_ints(h_str, self.general_to_i))

        # (4) B layer: 19-16-,28-25-,36-33-... -> dict
        b_str = self.df['b'][idx]
        b_vec = torch.tensor(string_to_ints(b_str, self.general_to_i))

        # (5) T layer: 8-,9+,10+,11+,12+,...
        t_str = self.df['t'][idx]
        t_vec = torch.tensor(string_to_ints(t_str, self.t_to_i))

        # (6) M layer: {'', '0', '0m1', '1', '1m0'}
        m_str = self.df['m'][idx]
        if m_str == '0.0': m_str = '0'
        elif m_str == '1.0': m_str = '1'
        m_vec = torch.tensor(self.m_to_i[m_str])

        # (7) S layer: {'', '1'}
        s_str = self.df['s'][idx]
        s_vec = torch.tensor(self.s_to_i[s_str])

        # (8) I layer -> dict
        i_str = self.df['i'][idx]
        i_vec = torch.tensor(string_to_ints(i_str, self.general_to_i))

        vecs = (note_vec, c_vec, h_vec, b_vec, t_vec, m_vec, s_vec, i_vec)
        return image, vecs


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