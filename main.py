# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# im = Image.open('raw_data/ImpoundmentWBT_dam_height.tif')
# box = (10000, 10000, 10050, 10050)
# croppedImage = im.crop(box)
# croppedImage.save("croppedImage_impoundment.tif")

# %%
im = Image.open("croppedImage_impoundment.tif")
arr = np.array(im)
for i in arr:
    for j in i:
        print(round(j, 3), end="  ")
    print("\n")
