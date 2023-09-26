# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 16:00:07 2023

@author: Niall

1. Read in stack of geometric images.
2. Set treshold to 90% max HU.
3. Find sum of HU in each slice.
4. Plot sum of HU in each slice:
       Peaks should correspond to bead centres.
5. Set a new threshold to find the slices where the centres of the beads are.
6. Multiply by slice width:
       Distance in z-direction.
"""

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Read in an image
path = str(input('input the file path:    '))
dicom_stack = pydicom.dcmread(path)

# Convert stack to a pixel array
# This isolates just the numerical data (no dicom headers)
stack = dicom_stack.pixel_array


flattened = [item for sublist in stack for item in sublist]

#Find the maximum and threshold
current_max = max(flattened[0])

for i in range(0, len(flattened)):
    if max(flattened[i]) > current_max:
        current_max = max(flattened[i])
        
thresh = int(current_max*0.97)

threshd_stack = []

for i in stack: #slice
    threshd_stack.append(np.where(i > thresh, i, 0))

plt.figure(1)
plt.imshow(threshd_stack[200])
plt.title('Bead only')
plt.axis('off')

#Find the peaks
areas = []

for i in threshd_stack:
    area = np.sum(i)
    areas.append(area)
    
#Set up slice index
index = [i for i in range(0,len(areas))]




thresh2 = max(areas)*0.97 
#Thresholding a second time to find the centre of each bead
points = []
for i in areas: #slice
    points.append(np.where(i > thresh2, i, 0)/thresh2)


plt.figure(2)   
plt.plot(index, points) 
plt.title('Profile of slices with beads')
plt.xlabel('Slice')
plt.ylabel('Bead peaks')

#Convert slices to distance
slices = []
for i in range(0, len(points)):
    if points[i] != 0:
        slices.append(i)
        
thickness = 0.2 #mm

vals = set(map(lambda x: int(str(x)[1]), slices))

#Set up a list with the peaks
indices = []

for sample in slices:
    if len(indices) == 0:
        indices.append(sample)
    else:
        if indices[len(indices) - 1] < sample-10:
            indices.append(sample)
        else:
            continue
#print('indices', indices)
#Multiply the distance (in pixels) between the peaks by 
#the slice thickness to get a physical distance

dists = []
for i in indices:
    d = round(i*thickness,2)
    dists.append(d)



output = pd.DataFrame(indices, dists)
print(output)
