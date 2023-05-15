from typing import Optional, Tuple, Dict
from pathlib import Path
import numpy as np
import torch
import imageio.v3 as iio
from PIL import Image
import cv2
from torch.utils.data import Dataset
import os
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def load_img(path: str) -> np.array:
    return iio.imread(path)

def load_scaled_depth(path: str) -> np.array:
    """
    Loads the scaled depth map from the given path (near: 0 - far: 1)
    """
    disparity = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    disparity = Image.fromarray(disparity)
    disparity = np.asarray(disparity, dtype=np.float32)
    return 1 - disparity

def resize_img(
        data: np.array, 
        dsize: tuple,
    ) -> np.array:
    data = cv2.resize(
        data, 
        dsize=dsize, 
        interpolation=cv2.INTER_NEAREST,
    )
    return data

def fix_scaled_depth_map(img: np.array) -> Tuple[np.array, np.array]:
    img = img.copy()
    mask = np.isfinite(img)
    img[~mask] = 1
    return img, mask

def get_filename_id(file_name: str) -> int:
    return int(file_name.split('_')[0])
    
def create_path_dict(dir_path: str) -> Dict[int, str]:
    path_dict = {}
    for file in os.listdir(dir_path):
        if file.endswith('.png') or file.endswith('.exr'):
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
        ) -> None:
        super().__init__()
        self.normalize = normalize
        self.dsize = dsize

        dataset_path = Path(dataset_path)
        if not dataset_path.isdir():
            raise ValueError(f"dataset path '{dataset_path}' is not a directory")
        
        self.image_dir_path = dataset_path / 'leftImg8bit'
        if not self.image_dir_path.isdir():
            raise ValueError(f"Image directory '{self.image_dir_path}' does not exist")

        self.label_dir_path = dataset_path / label_dir
        if not self.label_dir_path.isdir():
            logger.warning(f"Label directory '{self.label_dir_path}' does not exist. \
                           This is fine if you only intend to use this dataset for prediction.")
            self.label_dir_path = None

        self.image_path_dict = create_path_dict(self.image_dir_path)
        if self.label_dir_path is not None:
            self.label_path_dict = create_path_dict(self.label_dir_path)
            assert self.image_path_dict.keys() == self.label_path_dict.keys(), 'image and label path ids do not match'
        else:
            self.label_path_dict = None

        self.ids = np.array(list(self.image_path_dict.keys()))
        if shuffle_on_load:
            self.ids = np.random.permutation(self.ids)

    def _load_label(self, path: str) -> np.array:
        raise NotImplementedError("This method should be overridden by subclass")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        index_id = self.ids[index]
        image = load_img(self.image_path_dict[index_id])

        # if no label is available, return only the image
        if self.label_path_dict is None:
            if self.dsize is not None:
                image = resize_img(image, dsize=self.dsize)
            return dict(
                image=torch.tensor(image).permute(2, 0, 1).float(),
            )
        
        label = self._load_label(self.label_path_dict[index_id])
        if self.dsize is not None:
            image = resize_img(image, dsize=self.dsize)
            label = resize_img(label, dsize=self.dsize)

        # fill missing pixels in depth map
        if label.dtype == np.float32:
            label, mask = fix_scaled_depth_map(label)

        if self.normalize:
            image = image / 255.0

        return dict(
            image=torch.tensor(image).permute(2, 0, 1).float(),
            label=torch.tensor(label).unsqueeze(0),
            mask=torch.tensor(mask).unsqueeze(0),
        )
    
    def __len__(self) -> int:
        return len(self.ids)


class MUADSegmentationDataset(MUADBaseDataset):
    def __init__(
        self,
        dataset_path: str,
        dsize: Optional[tuple] = None,
        normalize: bool = True,
        shuffle_on_load: bool = False,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path, 
            dsize=dsize, 
            normalize=normalize, 
            shuffle_on_load=shuffle_on_load, 
            label_dir='leftLabel',
        )

    def _load_label(self, path: str) -> np.array:
        return load_img(path)
        

class MUADDepthDataset(MUADBaseDataset):
    """
    Loads the scaled depth map from the given path [near: 0,  far: 1]
    """
    def __init__(
        self,
        dataset_path: str,
        dsize: Optional[tuple] = None,
        normalize: bool = True,
        shuffle_on_load: bool = False,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path, 
            dsize=dsize, 
            normalize=normalize, 
            shuffle_on_load=shuffle_on_load, 
            label_dir='leftDepth',
        )

    def _load_label(self, path: str) -> np.array:
        return load_scaled_depth(path)
    
    @staticmethod
    def depth_to_meters(depth_map: np.array) -> np.array:
        return depth_map * 400.0
