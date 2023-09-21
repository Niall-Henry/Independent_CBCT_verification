# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:26:52 2023

@author: Niall
"""
#Load in necessary packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import pydicom


#Read in an image
path = str(input('input the file path:    '))
anatomy = input('Anatomy: ')
roi_size = 20
dicom_stack = pydicom.dcmread(path)

# Convert stack to a pixel array
# This isolates just the numerical data (no dicom headers)
stack = dicom_stack.pixel_array

# Select slice number:
n_slice = 100
 
phantom_slice = stack[n_slice, :, :]
plt.imshow(phantom_slice, cmap='gray')
plt.title('Original Image')

index = []
slices = [stack[i] for i in range(len(stack)) if i % 50 == 0]


num_slices = len(stack)

#start and end of the portion to be analysed and also how many slices to skip
start = int(num_slices * 0.1)
end = int(num_slices * 0.9)
slice_skip = 5

# Find the centre of the phantom
central_slice = stack[int(num_slices/2), :, :]

# Phantom diameter and pixel size used for initial guess of radius
phantom_diameter_mm = 80
pixel_size= 0.2
phantom_diameter_pixels = int(phantom_diameter_mm/pixel_size) 

# Find outline of the phantom
edges = canny(central_slice, sigma=5, low_threshold=5, high_threshold=20)

# Set a range of likely radii for the phantom
radius = np.arange((phantom_diameter_pixels//2-10), (phantom_diameter_pixels//2)+10, 1)
hough_res = hough_circle(edges, radius)

# Find the actual radius and the centre of the phantom for each slice
accums, cx, cy, radii = hough_circle_peaks(hough_res, radius, total_num_peaks=1)

print(f'(cx,cy) = ({cx}, {cy})')
for i in range(1, len(slices)-1):
    index.append(i*50)

#print(index)
#print(slices)

aluminium = []
air = []
pmma = []

aluminium_unc = []
air_unc = []
pmma_unc = []


plt.figure(1)
plt.imshow(slices[3],  cmap = 'bone')
plt.plot(cx, cy, 'o')
plt.plot(cx + 250, cy + 250, 'o')
plt.plot(cx + 100, cy + 100, 'o')

#Identify each region of interest in each slice, take the mean pixel value and
#convert to HU and also take the standard deviation (uncertainty).
for layer in slices[1:len(slices)]:
    #This code makes the assumption that the user has put the aluminium rod in
    #the central bore of the phantom.
    start_x_al = int(cx - roi_size//2)
    end_x_al = int(cx + roi_size//2)
    start_y_al = int(cy - roi_size//2)
    end_y_al = int(cy + roi_size//2)

    start_x_air = int(cx - roi_size//2)  + 250
    end_x_air = int(cx + roi_size//2) + 250
    start_y_air = int(cy - roi_size//2) + 250
    end_y_air = int(cy + roi_size//2) + 250

    start_x_pmma = int(cx - roi_size//2) + 100  
    end_x_pmma = int(cx + roi_size//2) + 100
    start_y_pmma = int(cy - roi_size//2) + 100
    end_y_pmma = int(cy + roi_size//2) + 100
    
    roi_air = layer[start_y_air:end_y_air, start_x_air:end_x_air ]
    roi_al = layer[start_y_al:end_y_al, start_x_al:end_x_al]
    roi_pmma = layer[start_y_pmma:end_y_pmma, start_x_pmma:end_x_pmma]
    
    #Taking the mean and converting to HU
    air.append(round((np.mean(roi_air) - 1024),2))
    aluminium.append(round((np.mean(roi_al) - 1024),2))
    pmma.append(round((np.mean(roi_pmma) - 1024),2))
    
    #Taking the uncertainty
    air_unc.append(round((np.std(roi_air) - 1024)**0.5,2))
    aluminium_unc.append(round((np.std(roi_al) - 1024)**0.5,2))
    pmma_unc.append(round((np.std(roi_pmma) - 1024)**0.5,2))
    

#Convert to arrays from lists to allow array operations and drop the last value
#to ensure there are no false readings due to phantom shape.    
indices_arr = np.array(index)    
aluminium_arr = np.array(aluminium[0:len(aluminium)-1])
air_arr = np.array(air[0:len(aluminium)-1])
pmma_arr = np.array(pmma[0:len(aluminium)-1])

aluminium_unc_arr = np.array(aluminium_unc[0:len(aluminium)-1])
air_unc_arr = np.array(air_unc[0:len(aluminium)-1])
pmma_unc_arr = np.array(pmma_unc[0:len(aluminium)-1])

#Plot the HU values along the length of the phantom
plt.figure(2)
plt.scatter(indices_arr, aluminium_arr, label = 'Aluminium', color = 'orange')
plt.scatter(indices_arr, pmma_arr, label = 'PMMA', color = 'green')
plt.scatter(indices_arr, air_arr, label = 'Air', color = 'blue')
# plt.errorbar(indices_arr, aluminium_arr, yerr= aluminium_unc_arr, xerr = None , color = 'orange', ls = None)
# plt.errorbar(indices_arr, pmma_arr, yerr = pmma_unc_arr, xerr = None, color = 'green')
# plt.errorbar(indices_arr, air_arr, yerr = air_unc_arr,  xerr = None, color = 'blue')
plt.legend(loc = 'upper right')
plt.xlabel('Slice')
plt.ylabel('HU')
plt.title(f'HU Value {anatomy}')

#Save values to a dataframe that can be output
dict_out = {'Slice': indices_arr, 'Aluminium': aluminium_arr, 'Air': air_arr, 'PMMA': pmma_arr}

DF = pd.DataFrame(dict_out)


print(DF)

#Estimate the mean and standard deviation of the pixel values for each material
al_mean = np.round(np.mean(aluminium_arr), 2)
air_mean = np.round(np.mean(air_arr), 2)
pmma_mean = np.round(np.mean(pmma_arr), 2)

al_std = np.round(np.std(aluminium_arr),2)
air_std = np.round(np.std(air_arr), 2)
pmma_std = np.round(np.std(pmma_arr), 2)

print('Al:  ', al_mean, '+-', al_std)
print('Air:  ', air_mean, '+-', air_std)
print('PMMA:  ', pmma_mean, '+-', pmma_std)