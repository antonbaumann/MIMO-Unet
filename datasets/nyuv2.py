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
            use_fraction: float = 1.0,
        ):
        super().__init__()
        h5_data = h5py.File(dataset_path, "r")
        self.data = dict(image=h5_data["image"], label=h5_data["depth"])
        self.normalize = normalize
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

        # label = 1 - label / 255.0 # convert disparity to depth

        label = label / 255.0

        if self.normalize:
            image = image / 255.0

        return {
            'image': torch.tensor(image).permute(2, 0, 1).float(),
            'label': torch.tensor(label).permute(2, 0, 1).float(),
        }

    def __len__(self):
        return self.num_items
    
    @staticmethod
    def depth_to_disparity(depth_map: np.array) -> np.array:
        return 1 - depth_map
