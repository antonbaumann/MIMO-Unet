from torch.utils.data import Dataset
import os
import numpy as np
import torch
import scipy.io
import scipy.ndimage
import cv2


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

def interpolate_depth_map(x, dsize: tuple):
    zoom_factor = (dsize[1] / x.shape[0], dsize[0] / x.shape[1])
    return scipy.ndimage.zoom(x, zoom_factor, order=3)

def load_images(base_path: str, image_paths: list, dsize=(345, 460)) -> np.array:
    images = []
    for image_path in image_paths:
        image = cv2.imread(os.path.join(base_path, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_img(image, dsize=dsize)
        images.append(image)
    return np.array(images)

def load_depth_maps(base_path: str, labal_paths: str, dsize=(345, 460)) -> np.array:
    depth_maps = []
    for label_path in labal_paths:
        data = scipy.io.loadmat(os.path.join(base_path, label_path))
        pos3dgrid = data['Position3DGrid']
        depth = pos3dgrid[:, :, 3]
        depth = interpolate_depth_map(depth, dsize=dsize)
        depth_maps.append(depth)
    return np.array(depth_maps)[..., np.newaxis]

class Make3dDepthDataset(Dataset):
    """
    Loads the Make3D dataset from the given path.
    The label is a scaled depth map (near: 0 - far: 1)
    """
    def __init__(
            self, 
            dataset_path: str,
            normalize: bool = True,
            shuffle_on_load: bool = False,
            use_fraction: float = 1.0,
        ):
        super().__init__()
        self.normalize = normalize

        image_paths = sorted([x for x in os.listdir(os.path.join(dataset_path, 'images')) if x.endswith('.jpg')])
        label_paths = sorted([x for x in os.listdir(os.path.join(dataset_path, 'labels')) if x.endswith('.mat')])

        images = load_images(os.path.join(dataset_path, 'images'), image_paths)
        labels = load_depth_maps(os.path.join(dataset_path, 'labels'), label_paths)

        if len (images) != len(labels):
            raise Exception('Number of images and labels must be equal. Got {} images and {} labels.'.format(len(images), len(labels)))
        
        masks = labels <= 70

        self.data = {
            'image': images,
            'label': labels,
            'mask': masks,
        }

        if shuffle_on_load:
            self.shuffle_permutation = np.random.permutation(len(self.data['image']))
        else:
            self.shuffle_permutation = np.arange(len(self.data['image']))
        
        if use_fraction < 1.0:
            self.num_items = int(len(self.data['image']) * use_fraction)
            self.shuffle_permutation = np.random.choice(self.shuffle_permutation, size=self.num_items, replace=False)
        else:
            self.num_items = len(self.data['image'])

    def __getitem__(self, index):
        shuffled_index = self.shuffle_permutation[index]
        image = self.data['image'][shuffled_index]
        label = self.data['label'][shuffled_index]
        mask = self.data['mask'][shuffled_index]

        if self.normalize:
            image = image / 255.0
            label = label / 120.0

        return {
            'image': torch.tensor(image).permute(2, 0, 1).float(),
            'label': torch.tensor(label).permute(2, 0, 1).float(),
            # 'mask': torch.tensor(mask).permute(2, 0, 1).float(),
        }

    def __len__(self):
        return self.num_items
