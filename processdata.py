from sklearn.preprocessing import MultiLabelBinarizer
from enum import unique
import rasterio
import numpy as np
from rasterio.windows import Window
from pathlib import Path
import os
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dataset import labels as unique_labels, label_names
import report

print = report.log
file_type = 'tiff'

class ProcessData:
    def __init__(self, data_dir = "data/deepcrop/tiles/", bands=["GRN", "NIR", "RED"], times=range(0,36,1)):
        self.data_dir = data_dir
        self.imageWidth = 224
        self.imageHeight = 224
        self.window_step = 200

        self.times = times
        self.bands = bands


    def process_tiles(self, tiles, data_filename = '2018-2018_001-365_HL_TSA_SEN2L_{band}_TSI.tif', label_filename = '/IACS_2018.tif', out_dir = 'data/prepared/'):
        bands = self.bands
        imageWidth = self.imageWidth
        imageHeight = self.imageHeight
        window_step = self.window_step

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_filename = '{tile}_{x}_{y}_{band}.tif'

        files = []

        for tile in tiles:           
            for band_i, band in enumerate(self.bands):   
                file_path = os.getcwd()+"/"+self.data_dir+tile+"/"+data_filename.format(band = band)
                print("processing: " + file_path)
                with rasterio.open(file_path) as src:
                    for x in np.arange(0, src.width, window_step):
                        for y in np.arange(0, src.height, window_step):              
                            
                            # print(f'Reading Window: {x},{y}')
                            window = Window(x,y,imageWidth, imageHeight)
                            w = src.read(range(1,37,1), window=window)

                            # write meta data    
                            xform = rasterio.windows.transform(window, src.meta['transform'])    
                            meta_d=src.meta.copy()    
                            meta_d.update({"height": imageWidth,
                                        "width": imageHeight,
                                        "transform": xform})

                            filename = '{tile}_{x}_{y}'.format(tile=tile, x=x, y=y)
                            if band_i == 0: files.append(filename)

                            outfile_path = out_dir+out_filename.format(tile=tile, x=x, y=y, band = band)
                            with rasterio.open(outfile_path, "w", **meta_d) as dest:
                                dest.write(w)

        # Class labels
        print("Reading Class labels")
        labels = []
        for tile in tiles:
            label_file = os.getcwd()+"/"+ self.data_dir + tile+"/" + label_filename
            with rasterio.open(label_file) as src:
                #window stepping
                for x in np.arange(0, src.width, window_step):
                    for y in np.arange(0, src.height, window_step):  
                        window = Window(x,y,imageWidth, imageHeight)
                        w = src.read(1, window=window)
                        sample_labels, sample_label_counts = np.unique(w, return_counts=True)
                        # Remove label that only occurs in 0.2 percent of the image
                        sample_label_mask = sample_label_counts > ((w.shape[0] * w.shape[1])/500)
                        labels.append(list(sample_labels[sample_label_mask]))
                  

        #image ids dataframe
        image_files = pd.DataFrame(data={'image_file': files})
    
        mlb = MultiLabelBinarizer(classes=unique_labels)
        one_hot_labels = mlb.fit_transform(labels)
        one_hot_df = pd.DataFrame(one_hot_labels,columns=unique_labels)

        one_hot_df = image_files.join(one_hot_df)
        one_hot_df.to_csv(out_dir + "labels.csv", index=True)           


    def read_dataset(self,  data_dir, t_samples=False):
        label_df = pd.read_csv(os.getcwd()+"/"+data_dir+"/labels.csv",index_col= 0)
        sample_n = label_df.shape[0]

        filenames = label_df["image_file"]

        data = np.empty((sample_n, len(self.bands), len(self.times), self.imageWidth, self.imageHeight), dtype=float)
        for i, filename in enumerate(filenames):
            for band in self.bands:
                filename_band = f'{filename}_{band}.tif'
                band_i = self.bands.index(band)
                with rasterio.open(os.path.join(data_dir, filename_band)) as src:
                    data[i, band_i] = src.read()[self.times, :, :]
        
        #labels
        unique_labels = list(filter(lambda c: c not in ["image_file"], label_df.columns))
        #Sample Selection
        if not t_samples: t_samples = sample_n

        labels = label_df[unique_labels].to_numpy()[:t_samples]
        data = data[0:t_samples, :]

        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).float()

        return data, labels
    

    def train_test_val_split(self, data, labels, test_split, val_split): 
    
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_split, random_state=42)
        X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=val_split, random_state=42)


        return X_train, X_test, X_val, y_train, y_test, y_val


    #def data_labels(data): 
    