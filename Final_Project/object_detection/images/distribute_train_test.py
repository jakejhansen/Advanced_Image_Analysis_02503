import glob
import numpy as np
from shutil import copy2
images = glob.glob('*.png')

print("Copying images")
for img in images:
    if np.random.rand() < 0.85:
        copy2(img, 'train/')
        copy2(img[:-4] + '.xml', 'train/')
    else:
        copy2(img, 'test/')
        copy2(img[:-4] + '.xml', 'test/')

print("Done")