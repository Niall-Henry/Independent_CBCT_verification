# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:39:30 2023

@author: Niall
"""

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks






#Read in an image
path = str(input('input the file path:    '))
dicom_stack = pydicom.dcmread(path)

# Convert stack to a pixel array
# This isolates just the numerical data (no dicom headers)
stack = dicom_stack.pixel_array

# Select and display slice 100
n_slice = 100

phantom_slice = stack[n_slice, :, :]

plt.figure(1)
plt.imshow(phantom_slice, cmap='gray')
plt.title('Original Image')

#Take samples of the phantom without the bead
samples = [stack[100], stack[250], stack[350], stack[450], stack[500]]
names = ['100', '250', '350', '450', '500']

#Find the outer edge of each of the phantom slices
params = []


for n_slice in samples:
    # Find outline of the phantom
    edges = canny(n_slice, sigma=5, low_threshold=5, high_threshold=20)

    # Set a range of likely radii for the phantom
    radius = np.arange(150, 250, 5)
    hough_res = hough_circle(edges, radius)

    # Find the actual radius and the centre of the phantom for each slice
    accums, cx, cy, radii = hough_circle_peaks(hough_res, radius, total_num_peaks=1)
    
    circle = [accums, cx, cy, radii]
    params.append(circle)
    
#Plot a slice of the phantom and overlay the circle
outline = plt.Circle((cx,cy),radii, color = 'r')


plt.figure()
plt.imshow(samples[0])
plt.gca().add_patch(outline)
plt.title('Outer ROI')

circles = [('x','y','r')]

for i in params:
    x = int(i[1])
    y = int(i[2])
    r = int(i[3])
    
    circle_params = (x,y,r)
    circles.append(circle_params)
    
#Make a list of the diameters and convert to mm
outer_diams = []
j = 0
for index in range(0, len(circles)):
    if index == 0:
        continue
        
    else:
        outer_diams.append(int(circles[index][2])*2)


outer_diams = np.array(outer_diams)


#Find the inner edge of each of the slices
inner_params = []

for slice in samples:
    # Find outline of the phantom
    edges = canny(slice, sigma=5, low_threshold=5, high_threshold=20)

    # Set a range of likely radii for the phantom
    radius = np.arange(100, 150, 5)
    hough_res = hough_circle(edges, radius)

    # Find the actual radius and the centre of the phantom for each slice
    accums, cx, cy, radii = hough_circle_peaks(hough_res, radius, total_num_peaks=1)
    
    circle = [accums, cx, cy, radii]
    inner_params.append(circle)
    
#Display one of the edges
circle1 = plt.Circle((int(inner_params[0][1]),(int(inner_params[0][2]))),(int(inner_params[0][3])), color = 'r')


plt.figure()
plt.imshow(samples[0])
plt.gca().add_patch(circle1)
plt.title('Inner ROI')

inner_diams = []
j = 0
for i in range(0, len(inner_params)):
    inner_diams.append(int(inner_params[i][3])*2)

inner_diams = np.array(inner_diams)

#Convert to mm
inner_diams_mm = inner_diams*0.2
outer_diams_mm = outer_diams*0.2

#Merge inner and outer parameters into a dataframe
n_slice = [100, 250, 350, 450, 500]
labels = ['Slice', "Inner Diameter (mm, +/- 0.2)", 'Outer Diameter (mm, +/- 0.2)']

output = pd.DataFrame([n_slice, inner_diams_mm, outer_diams_mm], labels)

print(output)


#Plot a sample with the fitted circles overlaid
fig, ax = plt.subplots(1, 3)
for a in ax:
    a.axis('off')

inside = plt.Circle((cx,cy),radii, color = 'r')
outside = plt.Circle((circles[1][0], circles[1][1]), circles[1][2], color = 'r')
ax[0].imshow(samples[len(samples)-1])
ax[1].imshow(samples[len(samples)-1])
ax[2].imshow(samples[len(samples)-1])
ax[1].add_patch(inside)
ax[2].add_patch(outside)
ax[0].title.set_text('Slice')
ax[1].title.set_text('Fitted inner circle')
ax[2].title.set_text('Fitted outer circle')
