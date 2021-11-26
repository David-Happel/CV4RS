from sklearn.preprocessing import MultiLabelBinarizer
from enum import unique
import rasterio
import numpy as np
from rasterio.windows import Window
from matplotlib import pyplot
from pathlib import Path
import os
import re
import pandas as pd
import c3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class ProcessData:
    def __init__(self, data_dir, data_filename, out_dir):
        self.data_dir = data_dir
        self.data_filename = data_filename
        self.out_dir = out_dir
        self.sample_n = 225
        self.imageWidth = 224
        self.imageHeight = 224
        self.window_step = 200

        t_start = 1
        t_stop = 37
        t_step = 1
        self.times = range(t_start,t_stop,t_step)
        self.bands = ["GRN", "NIR", "RED"]
    

    def prepare_data(self, times, bands): 
        self.bands = bands
        self.times = times
        imageWidth = 224
        imageHeight = 224
        window_step = 200
        out_filename = '{sample}_{band}.tif'
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        #empty array to store new window image ids

        for band in self.bands:
            print("Band:", band)
            #open file    
            file_path = self.data_dir+self.data_filename.format(band = band)
            with rasterio.open(file_path) as src:
                print(src.count, src.width, src.height, src.crs) 
                print(src.meta)
                sample_i = 0
                for x in np.arange(0, src.width, window_step):
                    for y in np.arange(0, src.height, window_step):              
                        
                        print("Reading Window:", x,y)
                        window = Window(x,y,imageWidth, imageHeight)
                        w = src.read(times, window=window)

                        # write meta data    
                        xform = rasterio.windows.transform(window, src.meta['transform'])    
                        meta_d=src.meta.copy()    
                        meta_d.update({"height": imageWidth,
                                    "width": imageHeight,
                                    "transform": xform})
                
                        outfile_path = self.out_dir+out_filename.format(sample= sample_i, band = band)
                        with rasterio.open(outfile_path, "w", band=band, **meta_d) as dest:
                            dest.write_band(times, w)

                        sample_i += 1
        # Class labels
        file_path = 'data/deepcrop/tiles/X0071_Y0043/IACS_2018.tiff'

        #open file
        print("\n\n")
        print("Class labels")
        with rasterio.open(file_path) as src:
            print(src.count, src.width, src.height, src.crs) 
            print(src.meta)
            #window stepping
            labels = []

            for x in np.arange(0, src.width, window_step):
                for y in np.arange(0, src.height, window_step):  
                    window = Window(x,y,imageWidth, imageHeight)
                    w = src.read(1, window=window)
                    labels.append(list(np.unique(w)))
                  

        ids = range(sample_i)
        d = {'image_id': ids, 'labels': labels}
        df = pd.DataFrame(data=d)        
      
        #image ids dataframe
        image_ids = df["image_id"].to_frame()
    
        mlb = MultiLabelBinarizer()
        df1 = pd.DataFrame(mlb.fit_transform(df['labels']),columns=[0, 10, 31, 32, 33, 34, 41, 42, 43, 50, 60, 70, 91, 92, 100, 120, 140, 181, 182])
        df1 = image_ids.join(df1)
        df1.to_csv(self.out_dir + "labels.csv", index=True)


    def create_dataset(self, t_samples): 
        data = np.empty((self.sample_n, len(self.bands), len(self.times), self.imageWidth, self.imageHeight))
        for i, file in enumerate(os.listdir(self.out_dir)):
            if i > self.sample_n: break
            regex = re.search('(\d*)_(.*).tif', file)
            if not regex : continue
            sample = int(regex.group(1))
            band = regex.group(2)
            band_i = self.bands.index(band)
            if band_i < 0: print("band not found!")
            with rasterio.open(os.path.join(self.out_dir, file)) as src:
                data[sample, band_i] = src.read()
        

        label_df = pd.read_csv(self.out_dir+"labels.csv",index_col= 0)
        unique_labels = list(filter(lambda c: c not in ["image_id"], label_df.columns))
        
        #Sample Selection 
        labels = label_df[unique_labels].to_numpy()[:t_samples]
        data = data[0:t_samples, :]
        return data, labels
    

    def train_test_val_split(self, data, labels, test_split, val_split): 
    
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_split, random_state=42)
        X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=val_split, random_state=42)


        return X_train, X_test, X_val, y_train, y_test, y_val





        

        



        

    #def data_labels(data): 
    