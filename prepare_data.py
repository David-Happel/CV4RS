import rasterio
import numpy as np
from rasterio.windows import Window
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

imageWidth = 224
imageHeight = 224
window_step = 200

t_start = 1
t_stop = 37
t_step = 1

times = range(t_start,t_stop,t_step)
bands = ["GRN", "NIR", "RED"]

data_dir = "data/deepcrop/tiles/X0071_Y0043/"
data_filename = '2018-2018_001-365_HL_TSA_SEN2L_{band}_TSI.tif'

out_dir = "data/prepared/"
out_filename = '{sample}_{band}.tif'
Path(out_dir).mkdir(parents=True, exist_ok=True)
#empty array to store new window image ids

for band in bands:
    print("Band:", band)
    #open file    
    file_path = data_dir+data_filename.format(band = band)
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
file_path = 'data/deepcrop/tiles/X0071_Y0043/IACS_2018.tif'

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
            #print("Reading Window:", x,y)
            window = Window(x,y,imageWidth, imageHeight)
            w = src.read(1, window=window)
            labels.append(list(np.unique(w)))
            # print(list(np.unique(w)))




ids = range(sample_i)
print(len(labels))
print(len(ids))

d = {'image_id': ids, 'labels': labels}
df = pd.DataFrame(data=d)        
print(df.head(n=5))
print(df.loc[df['image_id'] == '0_NIR.tif'])

#image ids dataframe
image_ids = df["image_id"].to_frame()
#d = {'image_id': df["image_id"]}
#df_i = pd.DataFrame(data=d)  

mlb = MultiLabelBinarizer()
df1 = pd.DataFrame(mlb.fit_transform(df['labels']),columns=[0, 10, 31, 32, 33, 34, 41, 42, 43, 50, 60, 70, 91, 92, 100, 120, 140, 181, 182])
#df1 = df1.join(image_ids)
df1 = image_ids.join(df1)
df1.to_csv(out_dir + "labels.csv", index=True)


print(df1.head(n=5))