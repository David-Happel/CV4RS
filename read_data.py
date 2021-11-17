import rasterio
from rasterio.windows import Window
from matplotlib import pyplot

#open file    
with rasterio.open("data/deepcrop/tiles/X0071_Y0043/2018-2018_001-365_HL_TSA_SEN2L_GRN_TSI.tif") as src:
    print(src.count, src.width, src.height)  
    pyplot.imshow(src.read(1), cmap='pink')
    pyplot.figure()

    w = src.read(1, window=Window(0, 0, 224, 224))
    pyplot.imshow(w , cmap='pink')
    pyplot.figure()

    w = src.read(1, window=Window(200, 0, 224, 224))
    pyplot.imshow(w , cmap='pink')
    pyplot.figure()

    w = src.read(1, window=Window(400, 0, 224, 224))
    pyplot.imshow(w , cmap='pink')
    pyplot.show()

#write meta data    
# xform = rasterio.windows.transform(win, src.meta['transform'])    
# meta_d=src.meta.copy()    
# meta_d.update({"height": height,
#                "width": width,
#                "transform": xform})  
  
#write output    
# with rasterio.open(outfile, "w", **meta_d) as dest:                           
#     dest.write_band(1, w)