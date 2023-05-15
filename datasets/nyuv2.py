from torch.utils.data import Dataset
import numpy as np
import torch


class NYUv2DepthDataset(Dataset):
    def __init__(
            self, 
            data, 
            normalize: bool = True,
            shuffle_on_load: bool = False,
        ):
        super().__init__()
        self.data = data
        self.normalize = normalize
        if shuffle_on_load:
            self.shuffle_permutation = np.random.permutation(len(self.data['image']))
        else:
            self.shuffle_permutation = np.arange(len(self.data['image']))

    def __getitem__(self, index):
        shuffled_index = self.shuffle_permutation[index]
        image = self.data['image'][shuffled_index]
        label = self.data['label'][shuffled_index]

        if self.normalize:
            image = image / 255.0
            label = 1 - label / 255.0 # convert disparity to depth

        return {
            'image': torch.tensor(image).permute(2, 0, 1).float(),
            'label': torch.tensor(label).permute(2, 0, 1).float(),
        }

    def __len__(self):
        return len(self.data['image'])
