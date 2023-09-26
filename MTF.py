# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:41:21 2023

@author: Niall
"""

import time
import pydicom
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import sys


#----------------------------------------------------------------------------#
# Read in a stack of CNR images
#----------------------------------------------------------------------------#
path = str(input('Please input the file path for the stack with the MTF wire insert:    '))
dicom_stack = pydicom.dcmread(path)

#----------------------------------------------------------------------------#
# Timer for code execution
#----------------------------------------------------------------------------#
start_time = time.time()


#----------------------------------------------------------------------------#
# Convert stack to a pixel array
# This isolates just the numerical data (no dicom headers)
#----------------------------------------------------------------------------#
stack_unscaled = dicom_stack.pixel_array
stack_unscaled = stack_unscaled.astype('int32')


part = input('Input 1 for knee, 2 for finger, 3 for wrist, 4 for elbow:  ')

protocols = ['1','2','3','4']

if part not in protocols:
    print('Try again with a valid part')
    sys.exit()

#if part == protocols[0] or part == protocols[3]:
if part == protocols[0] or part == protocols[3]:
    
    stack_unscaled = dicom_stack.pixel_array
    stack_unscaled = stack_unscaled.astype('int32')
    stack_unscaled = stack_unscaled[::-1]
#----------------------------------------------------------------------------#
# Convert pixel array to Hounsfield Units
#----------------------------------------------------------------------------#
stack = stack_unscaled-1024


#----------------------------------------------------------------------------#
# How many images are in the stack?
#----------------------------------------------------------------------------#
num_slices = len(stack)
 

#----------------------------------------------------------------------------#
# We want to analyse the central 20 % of the total number of slices
# NOTE: USE WIDER RANGE IN REAL ANALYSIS
#----------------------------------------------------------------------------#
slice_start = int(num_slices*0.7)
slice_end = int(num_slices*0.8)


#----------------------------------------------------------------------------#
# Define the number of slices to skip each time
# The higher the number, the quicker the analysis
#----------------------------------------------------------------------------#
slice_skip = 5

#----------------------------------------------------------------------------#
# Plot a slice with the wire
#----------------------------------------------------------------------------#
first_slice = stack[480, :, :]

plt.figure(1)
plt.title('Test image')
plt.imshow(first_slice)


#----------------------------------------------------------------------------#
# Phantom diameter and pixel size used for initial guess of radius
#----------------------------------------------------------------------------#
phantom_diameter_mm = 80
pixel_size_mm= 0.2
slice_thickness_mm = 0.2
phantom_diameter_pixels = int(phantom_diameter_mm/pixel_size_mm) 


#----------------------------------------------------------------------------#
# Find outline of the phantom
#----------------------------------------------------------------------------#
edges = canny(first_slice, sigma=5, low_threshold=5, high_threshold=20)


#----------------------------------------------------------------------------#
# Set a range of likely radii for the phantom
#----------------------------------------------------------------------------#
radius = np.arange((phantom_diameter_pixels//2-10), (phantom_diameter_pixels//2)+10, 1)
hough_res = hough_circle(edges, radius)


#----------------------------------------------------------------------------#
# Find the actual radius and the centre of the phantom for each slice
#----------------------------------------------------------------------------#
accums, cx, cy, radii = hough_circle_peaks(hough_res, radius, total_num_peaks=1)


roi_size = 80

start_x = int(cx - roi_size//2)
end_x = int(cx + roi_size//2)
start_y = int(cy - roi_size//2)
end_y = int(cy + roi_size//2)



#----------------------------------------------------------------------------#
# START OF LOOP FOR MTF CALCULATION
#----------------------------------------------------------------------------#
    

slice_used=[]
corrected_lsf_list = []

n = 480
# For each slice in the defined range
for slices in range(slice_start, slice_end):
    
    
    # Read in individual slices of the stack
    wire_im = stack[slices, :, :]
 
    #-------------------------------------------------------------------------#
    if slices % slice_skip == 0:
    
        # Track the slice being used
        slice_used = np.append(slice_used, [slices], axis=0)
        # select the region of interest using NumPy indexing
        roi = wire_im[start_y:end_y, start_x:end_x]
    
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        plt.title(f'Slice {n}')
        ax[0].imshow(wire_im, cmap='gray')
        ax[0].plot([cx], [cy], marker='o', markersize=2, color="red")
        ax[1].imshow(roi)
        
        impulse_location = np.where(roi == np.max(roi))
        impulse_roi = roi[impulse_location[0][0]-16:impulse_location[0][0]+16, impulse_location[1][0]-16:impulse_location[1][0]+16]
        
        #plt.figure()
        #plt.imshow(impulse_roi)
        
        #background mean
        background = roi[(roi.shape[0]//2-16):roi.shape[0]//2+16, (roi.shape[1]//2-16):roi.shape[1]//2+16]
        bg_mean = np.mean(background)
        
        impulse_sub_bg = impulse_roi - bg_mean
        
        #plt.figure()
        #plt.imshow(impulse_sub_bg)
        
        #want the average lsf in the vertical and horizontal direction. Should be symmetrical
        horizontal_lsf = np.sum(impulse_sub_bg, axis=0)
        try:
            average_lsf = np.mean([np.sum(impulse_sub_bg, axis=1),horizontal_lsf], axis=0)
        
        except:
            ValueError
            average_lsf = horizontal_lsf
        
        #Zero'ing the impulse and normalising.
        L = 5 #area outside LSF
        mean_1 = np.mean(average_lsf[0:L])
        mean_2 = np.mean(average_lsf[-L:])
        mean_means = np.mean([mean_1, mean_2])
        mean_sub_lsf = average_lsf - mean_means
        corrected_lsf = mean_sub_lsf / np.sum(mean_sub_lsf)
        
        plt.figure()
        plt.plot(horizontal_lsf)
        plt.title(f'Horizontal LSF, Slice {n}')
        
        plt.figure()
        plt.plot(average_lsf)
        plt.title(f'Average LSF, Slice {n}')
        
        plt.figure()
        plt.plot(corrected_lsf)
        plt.title(f'Corrected LSF, Slice {n}')
        
        corrected_lsf_list.append(corrected_lsf)
    
    n += 1
all_lsf = np.stack(corrected_lsf_list)
mean_lsf_all_slices = np.mean(all_lsf, axis=0)   
    
plt.figure()
plt.plot(mean_lsf_all_slices)
plt.title('Mean LSF from all slices')
plt.xlabel('Position (mm)')
plt.ylabel('Mean LSF')
    
    

pow2_for_norm = 32 #I don't know a good way to describe this, it's the next power of 2 after the length of the profile and is used to normalize the spatial frequency to the same range as the mtf

mtf = np.fft.fft(mean_lsf_all_slices, pow2_for_norm)
mtf_corrected = abs(mtf)

plt.figure()
plt.plot(mtf_corrected)
plt.title('Corrected MTF')
plt.xlabel('Spatial Frequency')
plt.ylabel('MTF')

pix_size = 0.2

sampling_freq = 1/pix_size
spatial_freq = np.arange(0, pow2_for_norm/2, 1) * sampling_freq/pow2_for_norm

#interpolate mtf with the spatial freq to then return results in more convenient range
new_range = np.arange(0, 2, 0.05) #in lp/mm
interp = interp1d(spatial_freq, mtf_corrected[0:pow2_for_norm//2]) #want to avoid U shape
new_mtf = interp(new_range)


plt.figure()
plt.plot(new_range, new_mtf, 'o')
plt.xlabel('Spatial Frequency (lp/mm)')
plt.ylabel('MTF')
plt.title(f'MTF wrt SF ({part})')
   
at_50 = np.where(new_mtf <= 0.5)[0][0]
at_10 = np.where(new_mtf <= 0.1)[0][0]
    
lp_at_50 = new_range[at_50]
lp_at_10 = new_range[at_10]

print('The limiting spatial resolution (MTF10) is: ', round(lp_at_10,2), ' lp/mm')   