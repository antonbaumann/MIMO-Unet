from typing import Optional, Tuple, Dict
import numpy as np
import torch
import imageio.v3 as iio
from PIL import Image
import cv2
from torch.utils.data import Dataset
import os


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
        self.image_dir_path = os.path.join(dataset_path, 'leftImg8bit')
        self.label_dir_path = os.path.join(dataset_path, label_dir)
        self.dsize = dsize

        self.image_path_dict = create_path_dict(self.image_dir_path)
        self.label_path_dict = create_path_dict(self.label_dir_path)
        
        assert self.image_path_dict.keys() == self.label_path_dict.keys(), 'image and label path ids do not match'

        self.ids = np.array(list(self.image_path_dict.keys()))
        if shuffle_on_load:
            self.ids = np.random.permutation(self.ids)

    def _load_label(self, path: str) -> np.array:
        raise NotImplementedError("This method should be overridden by subclass")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        index_id = self.ids[index]
        image = load_img(self.image_path_dict[index_id])
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
