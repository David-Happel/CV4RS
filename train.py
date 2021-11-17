import rasterio
import numpy as np
from rasterio.windows import Window
from matplotlib import pyplot
from pathlib import Path
import os
import re

bands = ["GRN", "NIR"]

data_dir = "data/prepared"

imageWidth = 224
imageHeight = 224
time_n = 36

sample_n = 225

data = np.empty((sample_n, len(bands), time_n, imageWidth, imageHeight))

for file in os.listdir(data_dir):
    with rasterio.open(os.path.join(data_dir, file)) as src:
        regex = re.search('(\d*)_(.*).tif', file)
        sample = int(regex.group(1))
        band = regex.group(2)
        band_i = bands.index(band)

        if band_i < 0: print("band not found!")

        data[sample, band_i] = src.read()

print(data.shape)
pyplot.imshow(data[10,1,5,:,:] , cmap='pink')
pyplot.show()
        

