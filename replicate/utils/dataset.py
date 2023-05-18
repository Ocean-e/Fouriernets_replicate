import numpy as np
import torch
import pandas as pd
import os
from torch.utils.data import Dataset

class DiffuserMirflickrDataset(Dataset):

    """
    Dataset for loading pairs of diffused images collected through DiffuserCam
    and ground truth, unblurred images collected through a DSLR camera. For use
    with DLMD (DiffuserCam Lensless Mirflickr Dataset). Optionally supports any
    callable transform.

    Args:
        csv_path: Path to .csv file containing filenames of images (both
        diffused and ground truth images share the same filename).

        data_dir: Path to directory containing diffused image data.

        label_dir: Path to directory containing ground truth image data.

        transform (optional): An optional callable that will be applied to
        every image pair. Defaults to None, in which case nothing happens.
    """

    def __init__(self, csv_path, data_dir, label_dir, transform=None):
        super().__init__()
        self.csv_contents = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_contents)

    def __getitem__(self, idx):

        img_name = self.csv_contents.iloc[idx, 0]

        path_diffused = os.path.join(self.data_dir, img_name)
        path_gt = os.path.join(self.label_dir, img_name)

        image = np.load(path_diffused+ ".npy") 
        label = np.load(path_gt + ".npy")
        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        sample = {"image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample