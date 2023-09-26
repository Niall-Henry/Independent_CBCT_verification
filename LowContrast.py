# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:22:08 2023

@author: Niall
"""


#Low contrast:
#1. 1000 ROIs
#2. Get mean pixel value for each ROI
#3. Get standard deviation of means
#4. Increase ROI size

# Import modules
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from math import isnan
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny

#Read in an image
path = str(input('Input the file path:    '))
dicom_stack = pydicom.dcmread(path)
Protocol = str(input('Anatomy:  '))

# Convert stack to a pixel array
# This isolates just the numerical data (no dicom headers)
stack = dicom_stack.pixel_array

# Select slice number:
n_slice = 100
 
# Display nth slice
plt.figure(1)

phantom_slice = stack[n_slice]
plt.imshow(phantom_slice, cmap='gray')
plt.title('Original Image')

phantom_diameter_mm = 80
pixel_size= 0.2
phantom_diameter_pixels = int(phantom_diameter_mm/pixel_size) 

# Find outline of the phantom
edges = canny(phantom_slice, sigma=5, low_threshold=5, high_threshold=20)

# Set a range of likely radii for the phantom
radius = np.arange((phantom_diameter_pixels//2-10), (phantom_diameter_pixels//2)+10, 1)
hough_res = hough_circle(edges, radius)

# Find the actual radius and the centre of the phantom for each slice
accums, cx, cy, radii = hough_circle_peaks(hough_res, radius, total_num_peaks=1)

    
# define the center point of the region of interest, taking the background
r1_x = cx + 100
r1_y = cy + 100
roi_size = 80

start_x = r1_x - roi_size//2
end_x = r1_x + roi_size//2
start_y = r1_y - roi_size//2
end_y = r1_y + roi_size//2



# select the first region of interest using NumPy indexing
roi1 = phantom_slice[int(start_y):int(end_y), int(start_x):int(end_x)]

r2_x = cx + 100
r2_y = cy - 100
roi_size = 80

start_x = r2_x - roi_size//2
end_x = r2_x + roi_size//2
start_y = r2_y - roi_size//2
end_y = r2_y + roi_size//2

# select the second region of interest using NumPy indexing
roi2 = phantom_slice[int(start_y):int(end_y), int(start_x):int(end_x)]

r3_x = cx - 100
r3_y = cy + 100
roi_size = 80

start_x = r3_x - roi_size//2
end_x = r3_x + roi_size//2
start_y = r3_y - roi_size//2
end_y = r3_y + roi_size//2

# select the third region of interest using NumPy indexing
roi3 = phantom_slice[int(start_y):int(end_y), int(start_x):int(end_x)]


r4_x = cx - 100
r4_y = cy - 100
roi_size = 80

start_x = r4_x - roi_size//2
end_x = r4_x + roi_size//2
start_y = r4_y - roi_size//2
end_y = r4_y + roi_size//2

# select the fourth region of interest using NumPy indexing
roi4 = phantom_slice[int(start_y):int(end_y), int(start_x):int(end_x)]

#Display each of the regions of interest

plt.figure(2)
plt.imshow(roi1, cmap='gray')
plt.title('Region of Interest 1')

plt.figure(3)
plt.imshow(roi2, cmap='gray')
plt.title('Region of Interest 2')

plt.figure(4)
plt.imshow(roi3, cmap='gray')
plt.title('Region of Interest 3')

plt.figure(5)
plt.imshow(roi4, cmap='gray')
plt.title('Region of Interest 4')


#Generate 100 pairs of coordinates where the image will be sampled from
coords = []

while len(coords) < 101:
    x = random.randint(0, 75)
    y = random.randint(0, 75)
    
    pair = (x,y)
    coords.append(pair)
    

#Set the sample sizes and put the regions of interest in a list that can be 
#iterated over
sample_size = [3,5,7,9,11,13,15,17,19]
rois = [roi1, roi2, roi3, roi4]

#Display the regions of interest and coordinate pairs
plt.figure(6)
plt.imshow(roi1, cmap='gray')
plt.title('Region of Interest 1')
for i in coords:
    plt.scatter(i[0], i[1])
    
plt.figure(7)
plt.imshow(roi2, cmap='gray')
plt.title('Region of Interest 2')
for i in coords:
    plt.scatter(i[0], i[1])
    
plt.figure(8)
plt.imshow(roi3, cmap='gray')
plt.title('Region of Interest 3')
for i in coords:
    plt.scatter(i[0], i[1])
    
plt.figure(9)
plt.imshow(roi4, cmap='gray')
plt.title('Region of Interest 4')
for i in coords:
    plt.scatter(i[0], i[1])
    
    
means = []

values = []

#Measure the mean pixel value for each ROI size at each coordinate
for j in sample_size:
    vals = []
    vals.append(j)
    for i in coords:
        start_x = i[0] - j//2
        end_x = i[0] + j//2
        start_y = i[1] - j//2
        end_y = i[1] + j//2
        
        #Select ROI and measure mean value (several times)
        for k in rois:
            block = k[start_x:end_x, start_y:end_y]
            if isnan(np.mean(block)) == False:
                vals.append(np.mean(block))
    
    #append array of values to values list
    values.append(np.array(vals))

#Calculate mean of means and append to new list
for i in values:
    means.append(np.mean(i))


vals_dict = {}
for i in values:
    j = 0
    vals_dict[round(i[j]*0.2,2)] = i[1::]
    j+=1

#Create a dictionary with the standard deviation and its uncertainty   
data_unc = {}
for i in vals_dict:
    data_unc[i] = (round(np.std(vals_dict[i])*3.29, 2), round(np.std(vals_dict[i])**0.5, 2))
    
size = list(data_unc.keys())
val = np.array([i[0] for i in data_unc.values()])
unc = [i[1] for i in data_unc.values()]
    
#Plot the standard deviation as a function of ROI size
plt.figure()
plt.errorbar(size, val, yerr = unc, xerr = None, fmt = 'o')
plt.xlabel('ROI size ($mm^{2}$)')
plt.ylabel('Low contrast limit (HU)')
plt.title(f'Low Contrast Detectability {Protocol}')
    

#Print the results for the user
dict_ser = pd.Series(data_unc)
print(dict_ser)

