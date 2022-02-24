from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import torch
# from torchvision.functional import normalize
import rasterio

# All labels
# label_names = ['Grassland', 'Winter Wheat', 'Winter Rye', 'Winter Barley', 'Other Winter Cereals', 'Spring Barley', 'Spring Oat', 'Other Spring Cereals', 'Winter Rapeseed', 'Legume', 'Sunflower',
#                   'Sugar Beet', 'Maize other', 'Maize for grain', 'Potato', 'Strawberry', 'Asparagus', 'Onion', 'Carrot', 'Other leafy Vegetables']
# labels = [10, 31, 32, 33, 34, 41, 42, 43, 50, 60, 70, 80, 91, 92, 100, 120, 130, 140, 181, 182]

# Selected Labels
label_names = ['Grassland', 'Winter Wheat', 'Winter Rye', 'Winter Barley', 'Other Winter Cereals', 'Spring Oat', 'Winter Rapeseed', 'Maize other', 'Legume']
labels = [10, 31, 32, 33, 34, 42, 50, 91, 60]

class DeepCropDataset(Dataset):
    def __init__(self, csv_file, root_dir, bands=["GRN", "NIR", "RED"], times=range(0,36,1), transform=None, t_samples=None):
        """Initialise dataset object. Custom crop dataset class that inherits the pytorch Dataset class  

        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            bands (list, optional): bands to be included in dataset. Defaults to ["GRN", "NIR", "RED"].
            times (list, optional): timepoints to be included in dataset . Defaults to range(0,36,1).
            transform (object, optional): transformations to be applied to data. Defaults to None.
            t_samples (int, optional): Number of samples dataset should be limited to. Defaults to None.
        """

        self.frame = pd.read_csv(os.path.join(root_dir,csv_file),index_col= 0)
        if t_samples: self.frame = self.frame.head(t_samples)

        self.root_dir = root_dir
        #instantiate class
        self.transform = transform

        self.bands = bands
        self.times = times

        self.imageWidth = 224
        self.imageHeight = 224
        self.labels = labels
        self.label_names = label_names

        self.label_counts = np.sum(self.frame.loc[:, map(str, self.labels)], axis=0)

    def __len__(self):
        """dataset length

        Returns:
            int: length of dataset
        """
        return len(self.frame)

    def __getitem__(self, idx):
        """ retrieves item from dataset given its id 

        Args:
            idx (int): data sample id

        Returns:
            tuple: datasample and labels
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        filename = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
        
        sample_data = np.empty((len(self.bands), len(self.times), self.imageWidth, self.imageHeight), dtype=float)
        for band_i, band in enumerate(self.bands):
            filename_band = f'{filename}_{band}.tif'
            with rasterio.open(filename_band) as src:
                sample_data[band_i] = src.read()[self.times, :, :]

        labels = self.frame.loc[idx, map(str, self.labels)].to_numpy(dtype=int)

        sample = (sample_data, labels)

        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Dataset Transform: Convert ndarrays in sample to Tensors. 
    """

    def __call__(self, sample):
        """Convert ndarrays in sample to Tensors.

        Args:
            sample (np.array, np.array): sample and labels for sample

        Returns:
            sample (Tensor, Tensor): sample and labels for sample as tensor
        """
        data, labels = sample
        return torch.from_numpy(data).float(), torch.from_numpy(labels).float()


class Normalise(object): 
    """Dataset Transform: "Normalise values for each band."""

    def __init__(self, mean, std):
        """Normalise values for each band

        Args:
            mean (list): mean value of bands
            std (list): standard deviation of bands
        """
        self.mean = mean
        self.std = std
        super().__init__()


    def __call__(self, sample):        
        """Normalise values for each band

        Args:
            sample (Tensor, Tensor): sample and labels for sample as tensor

        Returns:
            sample (Tensor, Tensor): sample and labels for sample normalized

        """
        data, labels = sample
        # For every channel, subtract the mean, and divide by the standard deviation
        for t, m, s in zip(data, self.mean, self.std):
            t.sub_(m).div_(s)
        return (data, labels)
    