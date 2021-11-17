import rasterio
import numpy as np
from rasterio.windows import Window
from pathlib import Path

imageWidth = 224
imageHeight = 224
window_step = 200

times = range(1,37)
bands = ["GRN", "NIR"]

data_dir = "data/deepcrop/tiles/X0071_Y0043/"
data_filename = '2018-2018_001-365_HL_TSA_SEN2L_{band}_TSI.tif'

out_dir = "data/prepared/"
out_filename = '{sample}_{band}.tif'

Path(out_dir).mkdir(parents=True, exist_ok=True)

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



  
