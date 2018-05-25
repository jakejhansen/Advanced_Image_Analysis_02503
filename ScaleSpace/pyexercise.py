# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 14:59:20 2018

@author: Jake
"""

#%% Exercise 2.1
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

def compute_g(t):
    t2 = np.ceil(5*np.sqrt(t))
    
    x = np.linspace(int(-t2/2), int(t2/2), int(t2))
    
    g = 1/np.sqrt(2*math.pi*t) * np.exp(-x**2 /(2*t))
    
    g_dd = (1/np.sqrt(2*math.pi*t) * np.exp(-x**2/(2*t)) * (x**2 - t)) / t**2
    
    g = g.reshape(g.shape[0], 1)
    g_dd = g_dd.reshape(g.shape[0], 1)
    
    return (g, g_dd)

(filt, fild_dd) = compute_g(100)

img = cv2.imread('test_blob_uniform.png', 0)

img_filtered = cv2.filter2D(img, -1, filt)

#plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(img, cmap=plt.cm.binary)
plt.subplot(1,2,2)
plt.imshow(img_filtered, cmap=plt.cm.binary)
plt.show()

#%% Exercise 2.2
from skimage.feature import peak_local_max
from skimage.filters.rank import maximum, minimum
from scipy import ndimage

img = cv2.imread('test_blob_uniform.png', 0)
img_res = img.copy()
img = img.astype('double')


t = 33

(filt, filt_dd) = compute_g(t)

Lxx = cv2.filter2D(img, -1, filt_dd)
Lxx = cv2.filter2D(Lxx, -1, filt.T)


Lyy = cv2.filter2D(img, -1, filt_dd)
Lyy = cv2.filter2D(Lyy, -1, filt.T)

L = Lxx + Lyy

coordinates = peak_local_max(L, min_distance=3, indices = False)
coordinates2 = peak_local_max(L*(-1), min_distance=3, indices = False)

coordinates[L < L.max()*0.7] = False
coordinates2[L > L.min()*0.7] = False

coordinates_max = np.argwhere(coordinates)
coordinates_min = np.argwhere(coordinates2)

img_res = cv2.cvtColor(img_res, cv2.COLOR_GRAY2RGB)
for cord in coordinates_max:    
    cv2.circle(img_res,(cord[1], cord[0]), int(math.sqrt(2*t)), (255,0,0), 1)
    
for cord in coordinates_min:    
    cv2.circle(img_res,(cord[1], cord[0]), int(math.sqrt(2*t)), (255,0,0), 1)
        

fig, axes = plt.subplots(1, 3, figsize=(16, 10), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()
ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(L, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('Maximum filter')


ax[2].imshow(img_res)
ax[2].autoscale(False)
ax[2].axis('off')
ax[2].set_title('Peak local max')

fig.tight_layout()
    
#%% Exercise 2.3

img = cv2.imread('test_blob_varying.png', 0)
img_res = img.copy()
img = img.astype('double')


scales = [10, 20, 40, 80, 160, 320, 500, 720, 900, 1200, 1500, 1800]

#Construct scale-space L
L = np.zeros((img.shape[0], img.shape[1], len(scales)))

for i, t in enumerate(scales):
    (filt, filt_dd) = compute_g(t)
    
    Lxx = cv2.filter2D(img, -1, filt_dd)
    Lxx = cv2.filter2D(Lxx, -1, filt.T)
    
    
    Lyy = cv2.filter2D(img, -1, filt_dd.T)
    Lyy = cv2.filter2D(Lyy, -1, filt)
    
    L_temp = Lxx + Lyy
    
    L[:,:,i] = t*L_temp
    


L_res_max = np.zeros_like(L)
L_res_min = np.zeros_like(L)


for i in range(1,L.shape[2]-1):
    print(i)
    for row in range(1,L.shape[0]-2):
        for col in range(1,L.shape[1]-2):
            subArray = L[row-1:row+2, col-1:col+2, i-1:i+2]
            v = subArray.flatten()
            v.sort()
            if L[row, col, i] > v[-2]:
                #L[row-1:row+2, col-1:col+2, i-1:i+2] = 0
                L_res_max[row, col, i] = 1
            if L[row, col, i] < v[1]:
                #L[row-1:row+2, col-1:col+2, i-1:i+2] = 0
                L_res_min[row, col, i] = 1
                

for i in range(1, L.shape[2]-1):
    L_res_max[L[:,:,i] < L.max()*0.85, i] = 0
    L_res_min[L[:,:,i] > L.min()*0.85, i] = 0

L = np.logical_or(L_res_min, L_res_max)
coordinates = np.argwhere(L)

img_res = cv2.cvtColor(img_res, cv2.COLOR_GRAY2RGB)
for cord in coordinates:    
    cv2.circle(img_res,(cord[1], cord[0]), int(math.sqrt(2*scales[cord[2]])), (255,0,0), 1)
    
fig, axes = plt.subplots(1, 4, figsize=(16, 10), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()
ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original')


ax[1].imshow(img_res)
ax[1].autoscale(False)
ax[1].axis('off')
ax[1].set_title('Peak local max')

ax[2].imshow(L[:,:,3]*255, cmap=plt.cm.gray)
ax[2].axis('off')
ax[2].set_title('Maximum filter')

ax[3].imshow(img_res)
ax[3].autoscale(False)
ax[3].axis('off')

fig.tight_layout()

#%% Exercise 2.4

img = cv2.imread('SEM.png', 0)
img = img[:650,:]
img[img < 150] = 0
img[img >= 150] = 255
img_res = img.copy()
img = img.astype('double')


scales = np.linspace(25,40,10)

#Construct scale-space L
L = np.zeros((img.shape[0], img.shape[1], len(scales)))

for i, t in enumerate(scales):
    t = int(t)
    
    (filt, filt_dd) = compute_g(t)
    
    Lxx = cv2.filter2D(img, -1, filt_dd)
    Lxx = cv2.filter2D(Lxx, -1, filt.T)
    
    
    Lyy = cv2.filter2D(img, -1, filt_dd.T)
    Lyy = cv2.filter2D(Lyy, -1, filt)
    
    L_temp = Lxx + Lyy
    
    L[:,:,i] = t*L_temp
    


L_res_max = np.zeros_like(L)
L_res_min = np.zeros_like(L)

Lb = L.copy()

for i in range(1,L.shape[2]-1):
    print(i)
    for row in range(1,L.shape[0]-2):
        for col in range(1,L.shape[1]-2):
            subArray = L[row-1:row+2, col-1:col+2, i-1:i+2]
            v = subArray.flatten()
            v.sort()
            if L[row, col, i] > v[-2]:
                #L[row-1:row+2, col-1:col+2, i-1:i+2] = 0
                L_res_max[row, col, i] = 1
            if L[row, col, i] < v[1]:
                L[row-1:row+2, col-1:col+2, i-1:i+2] = L[row, col, i]
                L_res_min[row, col, i] = 1
                

for i in range(1, L.shape[2]-1):
    L_res_max[Lb[:,:,i] < Lb.max()*0.4, i] = 0
    L_res_min[Lb[:,:,i] > Lb.min()*0.4, i] = 0

L = L_res_min
coordinates = np.argwhere(L)

img_res = cv2.cvtColor(img_res, cv2.COLOR_GRAY2RGB)
for cord in coordinates:    
    cv2.circle(img_res,(cord[1], cord[0]), int(math.sqrt(2*scales[cord[2]])), (255,0,0), 1)
    
fig, axes = plt.subplots(1, 3, figsize=(16, 10), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()
ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original')


ax[2].imshow(img_res)
ax[2].autoscale(False)
ax[2].axis('off')
ax[2].set_title('Peak local max')

L_disp = np.zeros((L.shape[0],L.shape[1]))
for i in range(L.shape[2]):
    L_disp = np.logical_or(L_disp, L[:,:,i])

ax[1].imshow(L_disp*255, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('Maximum filter')

fig.tight_layout()