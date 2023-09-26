# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 09:07:55 2023

@author: davidcaldwell
"""

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import time
from scipy.optimize import curve_fit
from skimage.draw import circle_perimeter
from skimage import color
from scipy.ndimage.filters import convolve

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
ax.plot([cx], [cy], marker='o', markersize=1, color="red")
plt.show()
    




#----------------------------------------------------------------------------#
# Plot original image with the outline of the target overlaid
#----------------------------------------------------------------------------#
fig, ax = plt.subplots(ncols=1, nrows=1)
image = color.gray2rgb(central_slice)
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius, shape=central_slice.shape)
    
    # Set RGB colour of the circle
    image[circy, circx] = (0, 255, 0)

ax.imshow(image)
plt.show()


#----------------------------------------------------------------------------#
# START OF LOOP FOR TTF CALCULATION
#----------------------------------------------------------------------------#
    

slice_used=[]
ttf_all_slices_list =[]
ttf_10_all_slices =[]

# For each slice in the defined range
for slices in range(slice_start, slice_end):
    
    # Read in individual slices of the stack
    cnr_slice = stack[slices, :, :]
         
    #-------------------------------------------------------------------------#
    if slices % slice_skip == 0:
    
        # Track the slice being used
        slice_used = np.append(slice_used, [slices], axis=0)
    
        # Region of interest for TTF in pixels
        roi_size = 50
        
        # The ROI contains one quadrant of the Al target plus the background       
        start_x = int(cx)
        end_x = int(cx + roi_size)
        start_y = int(cy)
        end_y = int(cy + roi_size)
        
        roi = cnr_slice[start_y:end_y, start_x:end_x]
        
        
        # Display the original image and an ROI using Matplotlib
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        # ax[0].imshow(cnr_slice, cmap='gray')
        # ax[0].add_patch(plt.Rectangle((cx, cy), roi_size, roi_size, linewidth=2, edgecolor='r', facecolor='none'))
        # ax[0].set_title('ROIs overlaid on CNR Slice')
        
        # ax[1].imshow(roi, cmap='gray')
        # ax[1].set_title('Central ROI')

        
        # 'dist' is a matrix of distances from the centre of the target for each pixel in 'roi'
        dist = np.zeros(roi.shape)    
        for j in range(roi.shape[0]):
            for k in range(roi.shape[1]):
                distance = np.sqrt(j**2+k**2)
                dist[j,k] = distance
                
        
        # Rearrange the pixel values and distances into columns
        # Also sorts the columns by distance from the origin            
        pv_flat = roi.flatten()
        dist_flat = dist.flatten()
        combined = np.vstack((dist_flat,pv_flat)).T
        sorted_col = combined[combined[:,0].argsort()]
        
        
        # This is the edge spread function, ESF
        # i.e., The pixel values are plotted as a functin of the distance from the origin.        
        dist_sorted = sorted_col[:,0]
        pv_sorted = sorted_col[:,1]
        
        # Slight smoothing of ESF
        esf_region = pv_sorted
        smoothing_filter = [0.25, 0.5, 0.25]
        smooth_esf = convolve(esf_region,smoothing_filter)
        
        # Plot calculated ESF
        # plt.figure(figsize=(6,6))
        # plt.plot(dist_sorted,pv_sorted, 'co', label = 'Data')
        # plt.plot(dist_sorted,smooth_esf, 'b-', label = 'Smoothed')
        # plt.title('Edge Spread Function for Al target')
        # plt.legend(loc='best')
        # plt.xlabel('Pixels')
        # plt.ylabel('Hounsfield Units')     
        
        # ESF is noisy- fit sigmoid curve to it for smoother ESF      
        def sigmoid(x, L ,x0, k, b):
            y = L / (1 + np.exp(-k*(x-x0)))+b
            return (y)

        p0 = [max(smooth_esf), np.median(dist_sorted),2,min(smooth_esf)] # this is an mandatory initial guess
        
        popt, pcov = curve_fit(sigmoid, dist_sorted, smooth_esf,p0, method='dogbox')
        
        dist_fit = np.linspace(0, 60, 1000)
        esf_fit = sigmoid(dist_fit, *popt)
        
        # plt.figure(figsize=(6,6))
        # plt.plot(dist_sorted, smooth_esf, 'co', label='ESF (Smoothed)')
        # plt.plot(dist_fit,esf_fit, 'b-', label='Sigmoid Fit')
        # plt.legend(loc='best')
        # plt.title('Edge Spread Function for Al target')
        # plt.xlabel('Pixels')
        # plt.ylabel('Hounsfield Units')
                
        # Differentiate ESF to get LSF
        lsf = np.diff(esf_fit)
        lenLSF =len(lsf) 
        lsf = lsf/sum(lsf)
              
        # Plot LSF            
        # plt.figure(figsize=(6,6))
        # plt.plot(dist_fit[1:],lsf, 'r-')
        # plt.title('Line Spread Function for Al target')
        # plt.xlabel('Pixels')
                
        # Fourier Transform of LSF to get TTF
        ttf_step1= np.fft.fft(lsf)
        ttf_step2=abs(ttf_step1)/ttf_step1[0]
        ttf = np.real(ttf_step2)
         
        # Define spatial frequencies
        u = ((1/roi_size)*(1/pixel_size))*np.array(range(0,lenLSF))
        ny_u = 1/(2*pixel_size)
        
        # Spatial Frequencies up to Nyquist Frequency
        u_low = u[u <= ny_u]
        
        # TTF up to Nyquist Frequency
        ttf_useful = ttf[0:len(u_low)]
        
        # Plot TTF
        # plt.figure(figsize=(6,6))
        # plt.plot(u_low, ttf_useful, 'b-') 
        # plt.title('Task Transfer Function for Al target')
        # plt.xlabel('Spatial Frequency (lp $mm^{-1}$)')
        # plt.ylabel('TTF')
        
        # Calculate the TTF at 10 
        absol_val_array10 = np.abs(ttf_useful-0.1)
        smallest_difference_10 = absol_val_array10.argmin()
        closest_element10 = ttf_useful[smallest_difference_10]
        ind_10 = np.where(ttf_useful == closest_element10)
        lp_at_10 = round(float(u_low[ind_10]),2)
        print('The TTF10 value is ', lp_at_10, ' lp/mm for slice ', slices)
        
        
        # Save results to a list
        ttf_10_all_slices.append(lp_at_10)
        ttf_all_slices_list.append(ttf_useful)
        all_ttf = np.stack(ttf_all_slices_list)
        mean_ttf = np.mean(all_ttf, axis=0)
        mean_ttf_at_10 = round(np.mean(ttf_10_all_slices),2)



    

#----------------------------------------------------------------------------#
# Plot the TTF for each slice used
#----------------------------------------------------------------------------#
plt.figure(figsize=(6,6))
for slices_used in range(len(all_ttf)):
    plt.plot(u_low, all_ttf[slices_used][:], 'o', label = slice_used)
    plt.title('Task Transfer Function vs. Spatial Frequency (Per Slice)')
    plt.xlabel('Spatial Frequency (mm$^{-1}$)')
    plt.ylabel('TTF []')
    #plt.legend(loc='best')

#----------------------------------------------------------------------------#
# Plot the mean TTF for the phantom
#----------------------------------------------------------------------------#
plt.figure(figsize=(6,6))
plt.plot(u_low, mean_ttf, 'r-')
plt.title('Mean Task Transfer Function vs. Spatial Frequency (Overall)')
plt.xlabel('Spatial Frequency (mm$^{-1}$)')
plt.ylabel('TTF []')


#----------------------------------------------------------------------------#
# Print the time taken to run the script
#----------------------------------------------------------------------------#
print("--- %s seconds ---" % round((time.time() - start_time),2))  

print('The mean TTF10 value is ', mean_ttf_at_10, ' lp/mm.')

#----------------------------------------------------------------------------#
# Example Graph 1: Original Image + ROI 
#----------------------------------------------------------------------------#
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cnr_slice, cmap='gray')
ax[0].add_patch(plt.Rectangle((cx, cy), roi_size, roi_size, linewidth=2, edgecolor='r', facecolor='none'))
ax[0].set_title('ROIs overlaid on CNR Slice')
ax[1].imshow(roi, cmap='gray')
ax[1].set_title('Central ROI')


#----------------------------------------------------------------------------#
# Example Graph 2: Edge Spread Function
#----------------------------------------------------------------------------#
plt.figure(figsize=(6,6))
plt.plot(dist_sorted,pv_sorted, 'co', label = 'Data')
plt.plot(dist_sorted,smooth_esf, 'b-', label = 'Smoothed')
plt.title('Edge Spread Function for Al target')
plt.legend(loc='best')
plt.xlabel('Pixels')
plt.ylabel('Hounsfield Units')   


#----------------------------------------------------------------------------#
# Example Graph 3: Edge Spread Function- Sigmoid Fit
#----------------------------------------------------------------------------#
plt.figure(figsize=(6,6))
plt.plot(dist_sorted, smooth_esf, 'co', label='ESF (Smoothed)')
plt.plot(dist_fit,esf_fit, 'b-', label='Sigmoid Fit')
plt.legend(loc='best')
plt.title('Edge Spread Function for Al target')
plt.xlabel('Pixels')
plt.ylabel('Hounsfield Units')
  
              
#----------------------------------------------------------------------------#
# Example Graph 4: Line Spread Function
#----------------------------------------------------------------------------#          
plt.figure(figsize=(6,6))
plt.plot(dist_fit[1:],lsf, 'r-')
plt.title('Line Spread Function for Al target')
plt.xlabel('Pixels')
