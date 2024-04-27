import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os



def count_files_in_directory(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])



class HeatMapDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir = "/root/lisan-ai-gaib/AaltoAI-Hackathon-LaG/data", transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return count_files_in_directory(f"{self.root_dir}/hack")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        else:
            idx = [idx]
        for id in idx:
            sample = torch.tensor(np.load(f'{self.root_dir}/hack/data_{id}.npy'))

        if self.transform:
            sample = self.transform(sample)

        return sample