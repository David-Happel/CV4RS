import rasterio
import numpy as np
from rasterio.windows import Window
from matplotlib import pyplot

imageWidth = 224
imageHeight = 224

bands = ["GRN", "NIR"]

for band in bands:
    print("Band:", band)
    #open file    
    with rasterio.open(f'data/deepcrop/tiles/X0071_Y0043/2018-2018_001-365_HL_TSA_SEN2L_{band}_TSI.tif') as src:
        print(src.count, src.width, src.height, src.crs) 
        for time in range(1, src.count+1):
            print("Time:", time)
            for x in np.arange(0, src.width, 200):
                for y in np.arange(0, src.height, 200):
                    print("Reading Window:", x,y)
                    window = Window(x,y,imageWidth, imageHeight)
                    w = src.read(time, window=window)

                    # write meta data    
                    xform = rasterio.windows.transform(window, src.meta['transform'])    
                    meta_d=src.meta.copy()    
                    meta_d.update({"height": imageWidth,
                                "width": imageHeight,
                                "transform": xform})  

                    # # write output    
                    # with rasterio.open(outfile, "w", **meta_d) as dest:                           
                    #     dest.write_band(band, w)


  
