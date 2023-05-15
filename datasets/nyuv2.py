from torch.utils.data import Dataset
import numpy as np
import torch
import h5py


class NYUv2DepthDataset(Dataset):
    """
    Loads the NYUv2 dataset from the given path.
    The label is a scaled depth map (near: 0 - far: 1)
    """
    def __init__(
            self, 
            dataset_path: str,
            normalize: bool = True,
            shuffle_on_load: bool = False,
        ):
        super().__init__()
        h5_data = h5py.File(dataset_path, "r")
        self.data = dict(image=h5_data["image"], label=h5_data["depth"])
        self.normalize = normalize
        if shuffle_on_load:
            self.shuffle_permutation = np.random.permutation(len(self.data['image']))
        else:
            self.shuffle_permutation = np.arange(len(self.data['image']))

    def __getitem__(self, index):
        shuffled_index = self.shuffle_permutation[index]
        image = self.data['image'][shuffled_index]
        label = self.data['label'][shuffled_index]

        label = 1 - label / 255.0 # convert disparity to depth

        if self.normalize:
            image = image / 255.0

        return {
            'image': torch.tensor(image).permute(2, 0, 1).float(),
            'label': torch.tensor(label).permute(2, 0, 1).float(),
        }

    def __len__(self):
        return len(self.data['image'])
    
    @staticmethod
    def depth_to_disparity(depth_map: np.array) -> np.array:
        return 1 - depth_map
