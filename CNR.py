# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:58:56 2023

@author: Niall
"""

#Import modules
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import time
from skimage import color
from skimage.draw import circle_perimeter

#----------------------------------------------------------------------------#
# Read in a stack of CNR images
#----------------------------------------------------------------------------#
path = str(input('Please input the file path for the images with the Al insert:    '))
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
slice_start = int(num_slices*0.1)
slice_end = int(num_slices*0.9)


#----------------------------------------------------------------------------#
# Define the number of slices to skip each time
# The higher the number, the quicker the analysis
#----------------------------------------------------------------------------#
slice_skip = 25


#----------------------------------------------------------------------------#
# Find the centre of the insert
#----------------------------------------------------------------------------#
central_slice = stack[int(num_slices/2), :, :]


#----------------------------------------------------------------------------#
# Al Target diameter and pixel size used for initial guess of radius
#----------------------------------------------------------------------------#
target_diameter_mm = 10
pixel_size= 0.2
target_diameter_pixels = int(target_diameter_mm/pixel_size) 


#----------------------------------------------------------------------------#
# Find outline of the Al target
#----------------------------------------------------------------------------#
edges = canny(central_slice, sigma=10, low_threshold=20, high_threshold=50)

fig, ax = plt.subplots(ncols=1, nrows=1)
plt.imshow(edges, cmap='gray')


#----------------------------------------------------------------------------#
# Set a range of likely radii for the phantom
#----------------------------------------------------------------------------#
radius = np.arange((target_diameter_pixels//2-10), (target_diameter_pixels//2)+10, 1)
hough_res = hough_circle(edges, radius)


#----------------------------------------------------------------------------#
# Find the actual radius and the centre of the phantom for each slice
#----------------------------------------------------------------------------#
accums, cx, cy, radii = hough_circle_peaks(hough_res, radius, total_num_peaks=1)


#----------------------------------------------------------------------------#
# Plot original image with the centre of the target overlaid
#----------------------------------------------------------------------------#
fig, ax = plt.subplots(ncols=1, nrows=1)
image = color.gray2rgb(central_slice)
ax.imshow(image)
ax.plot([cx], [cy], marker='o', markersize=2, color="red")
plt.show()
    



#----------------------------------------------------------------------------#
# Plot original image with the outline of the target overlaid
#----------------------------------------------------------------------------#
fig, ax = plt.subplots(ncols=1, nrows=1)
image = color.gray2rgb(central_slice)
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius, shape=central_slice.shape)
    
    # Set RGB colour of the circle
    image[circy, circx] = (255, 0, 0)

ax.imshow(image)
plt.show()


#----------------------------------------------------------------------------#
# START OF LOOP FOR CNR CALCULATION
#----------------------------------------------------------------------------#
slice_used=[]
cnr_all_slices_list =[]

# For each slice in the defined range
for slices in range(slice_start, slice_end):
    
    # Read in individual slices of the stack
    cnr_slice = stack[slices, :, :]
         
    #-------------------------------------------------------------------------#
    if slices % slice_skip == 0:
    
        # Track the slice being used
        slice_used = np.append(slice_used, [slices], axis=0)
    
        # Region of interest for TTF in pixels
        roi_size = 32
        
        # Define the target region of interest (al_roi)       
        start_x1 = int(cx - roi_size/2)
        end_x1 = int(cx + roi_size/2)
        start_y1 = int(cy - roi_size/2)
        end_y1 = int(cy + roi_size/2)
        
        al_roi = cnr_slice[start_y1:end_y1, start_x1:end_x1]
        
        # Define the background region of interest (bkgd_roi)
        start_x2 = int(cx + (2*roi_size))
        end_x2 = int(cx + (3*roi_size))
        start_y2 = int(cy + (2*roi_size))
        end_y2 = int(cy + (3*roi_size))       
        
        bkgd_roi = cnr_slice[start_y2:end_y2, start_x2:end_x2]
        
        # Display the original image and an ROI using Matplotlib
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        
        ax[0].imshow(cnr_slice, cmap='gray')
        ax[0].add_patch(plt.Rectangle((start_x1, start_y1), roi_size, roi_size, linewidth=1, edgecolor='r', facecolor='none'))
        ax[0].add_patch(plt.Rectangle((start_x2, start_y2), roi_size, roi_size, linewidth=1, edgecolor='g', facecolor='none'))
        ax[0].set_title('ROI overlaid on CNR Slice')
        
        ax[1].imshow(al_roi, cmap='gray')
        ax[1].set_title('Al ROI')
        
        ax[2].imshow(bkgd_roi, cmap='gray')
        ax[2].set_title('Background ROI')
        

        # Calculate the mean HUs and st dev of HUs within the rois
        al_mean = np.mean(al_roi)
        bkgd_mean = np.mean(bkgd_roi)
        bkgd_stdev = np.std(bkgd_roi)
        
        # Calculate CNR
        CNR = abs(al_mean-bkgd_mean)/bkgd_stdev
        CNR = round(CNR,1)
        
        cnr_all_slices_list.append(CNR)
        

#----------------------------------------------------------------------------#
# Plot the CNR for each slice used
#----------------------------------------------------------------------------#
plt.figure(figsize=(6,6))
anatomy = input('Body part:  ')
cnr = round(np.mean(cnr_all_slices_list), 3)
unc = round(np.std(cnr_all_slices_list), 3)
for slices_used in range(len(cnr_all_slices_list)):
    plt.plot(slice_used, cnr_all_slices_list, 'o')
    plt.title(f"CNR (Per Slice, {anatomy})")
    plt.ylim(bottom=0, top =100)
    plt.xlabel('Slice Used')
    plt.ylabel('CNR')



print(f'CNR:   {cnr} +- {unc}')
#----------------------------------------------------------------------------#
# Print the time taken to run the script
#----------------------------------------------------------------------------#
print("--- %s seconds ---" % round((time.time() - start_time),2))  







































