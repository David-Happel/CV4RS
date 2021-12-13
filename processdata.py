from sklearn.preprocessing import MultiLabelBinarizer
from enum import unique
import rasterio
import numpy as np
from rasterio.windows import Window
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
    def __init__(self, data_dir = "data/deepcrop/tiles/", bands=["GRN", "NIR", "RED"], times=range(1,37,1)):
        self.data_dir = data_dir
        self.imageWidth = 224
        self.imageHeight = 224
        self.window_step = 200

        self.times = times
        self.bands = bands

    def process_tile(self, tile, data_filename = '2018-2018_001-365_HL_TSA_SEN2L_{band}_TSI.tif', label_filename = '/IACS_2018.tif', out_dir = 'data/prepared/'):
        bands = self.bands
        times = self.times
        imageWidth = self.imageWidth
        imageHeight = self.imageHeight
        window_step = self.window_step

        out_filename = '{sample}_{band}.tif'
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        #empty array to store new window image ids

        for band in self.bands:
            print("Band:", band)
            #open file    
            file_path = os.getcwd()+"/"+self.data_dir+tile+"/"+data_filename.format(band = band)
            print("processing: ", file_path)
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
                
                        outfile_path = out_dir+out_filename.format(sample= sample_i, band = band)
                        with rasterio.open(outfile_path, "w", band=band, **meta_d) as dest:
                            dest.write_band(times, w)

                        sample_i += 1
        
        # Class labels
        print("\n\n")
        print("Class labels")
        label_file = os.getcwd()+"/"+ self.data_dir + tile+"/" + label_filename
        with rasterio.open(label_file) as src:
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
        print(len(ids), len(labels))
        
        d = {'image_id': ids, 'labels': labels}
        df = pd.DataFrame(data=d)        
      
        #image ids dataframe
        image_ids = df["image_id"].to_frame()
    
        mlb = MultiLabelBinarizer()
        df1 = pd.DataFrame(mlb.fit_transform(df['labels']),columns=[0, 10, 31, 32, 33, 34, 41, 42, 43, 50, 60, 70, 91, 92, 100, 120, 140, 181, 182])
        df1 = image_ids.join(df1)
        df1.to_csv(out_dir + "labels.csv", index=True)      


    def read_dataset(self,  out_dir, t_samples=False):
        label_df = pd.read_csv(os.getcwd()+"/"+out_dir+"/labels.csv",index_col= 0)
        sample_n = label_df.shape[0]

        data = np.empty((sample_n, len(self.bands), len(self.times), self.imageWidth, self.imageHeight))
        for i, file in enumerate(os.listdir(os.getcwd()+"/"+out_dir)):
            regex = re.search('(\d*)_(.*).tif', file)
            if not regex : continue
            sample = int(regex.group(1))
            band = regex.group(2)
            band_i = self.bands.index(band)
            if band_i < 0: print("band not found!")
            with rasterio.open(os.path.join(out_dir, file)) as src:
                data[sample, band_i] = src.read()[self.times, :, :]
        
        #labels
        unique_labels = list(filter(lambda c: c not in ["image_id"], label_df.columns))
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
    