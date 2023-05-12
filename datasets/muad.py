from typing import Optional
import numpy as np
import torch
import imageio.v3 as iio
from PIL import Image
import cv2
from torch.utils.data import Dataset
import os
from tqdm import tqdm


def load_img(path: str) -> np.array:
    data = iio.imread(path)
    return data

def load_depth(path: str, to_meters: bool) -> np.array:
    depth = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth = Image.fromarray(depth)
    depth = np.asarray(depth, dtype=np.float32)
    depth = 1 - depth

    if to_meters:
        depth = depth * 400

    return depth

def resize_img(
        data: np.array, 
        dsize: tuple,
    ) -> np.array:
    data = cv2.resize(
        data, 
        dsize=dsize, 
        interpolation=cv2.INTER_CUBIC
    )
    return data

def get_filename_id(file_name: str) -> int:
    return int(file_name.split('_')[0])
    
def create_path_dict(dir_path: str) -> dict:
    path_dict = {}
    print(f'Scanning all files in {dir_path}')
    for file in tqdm(os.listdir(dir_path)):
        if file.endswith('.png'):
            file_id = get_filename_id(file)
            path_dict[file_id] = os.path.join(dir_path, file)
    return path_dict


class MUADBaseDataset(Dataset):
    def __init__(
            self,
            dataset_path: str,
            dsize: Optional[tuple] = None,
            normalize: bool = True,
            shuffle_on_load: bool = False,
            label_dir: str = '',
        ):
        super().__init__()
        self.normalize = normalize
        self.image_dir_path = os.path.join(dataset_path, 'leftImg8bit')
        self.label_dir_path = os.path.join(dataset_path, label_dir)
        self.dsize = dsize

        self.image_path_dict = create_path_dict(self.image_dir_path)
        self.label_path_dict = create_path_dict(self.label_dir_path)
        
        common_keys = set(self.image_path_dict.keys()).intersection(set(self.label_path_dict.keys()))
        if len(common_keys) != len(self.image_path_dict.keys()):
            print(f'Warning: {len(self.image_path_dict.keys()) - len(common_keys)} image files do not have corresponding label files')
        if len(common_keys) != len(self.label_path_dict.keys()):
            print(f'Warning: {len(self.label_path_dict.keys()) - len(common_keys)} label files do not have corresponding image files')

        assert self.image_path_dict.keys() == self.label_path_dict.keys(), 'image and label path ids do not match'

        self.ids = np.array(list(self.image_path_dict.keys()))
        if shuffle_on_load:
            self.ids = np.random.permutation(self.ids)

    def _load_label(self, path):
        raise NotImplementedError("This method should be overridden by subclass")

    def __getitem__(self, index):
        index_id = self.ids[index]
        image = load_img(self.image_path_dict[index_id])
        
        if self.dsize is not None:
            image = resize_img(image, dsize=self.dsize)

        if self.normalize:
            image = image / 255.0

        label = self.load_label(self.label_path_dict[index_id])

        return dict(
            image=torch.tensor(image).permute(2, 0, 1).float(),
            label=label,
        )
    
    def __len__(self):
        return len(self.ids)


class MUADSegmentationDataset(MUADBaseDataset):
    def __init__(
        self,
        dataset_path: str,
        dsize: Optional[tuple] = None,
        normalize: bool = True,
        shuffle_on_load: bool = False,
    ):
        super().__init__(
            dataset_path=dataset_path, 
            dsize=dsize, 
            normalize=normalize, 
            shuffle_on_load=shuffle_on_load, 
            label_dir='leftLabel',
        )

    def _load_label(self, path):
        label = load_img(path)
        return torch.tensor(label).unsqueeze(0).long()


class MUADDepthDataset(MUADBaseDataset):
    def __init__(
        self,
        dataset_path: str,
        dsize: Optional[tuple] = None,
        normalize: bool = True,
        shuffle_on_load: bool = False,
    ):
        super().__init__(
            dataset_path=dataset_path, 
            dsize=dsize, 
            normalize=normalize, 
            shuffle_on_load=shuffle_on_load, 
            label_dir='leftDepth',
        )

    def _load_label(self, path):
        label = load_depth(path, to_meters=not self.normalize)
        return torch.tensor(label).unsqueeze(0).float()
