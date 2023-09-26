# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:42:10 2023

@author: davidcaldwell
"""
#----------------------------------------------------------------------------#
# Import relevant modules
#----------------------------------------------------------------------------#
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import time
from scipy.optimize import curve_fit
import warnings
from scipy.signal import savgol_filter as sg
from skimage.draw import circle_perimeter
from skimage import color


#----------------------------------------------------------------------------#
# Timer for code execution
#----------------------------------------------------------------------------#
start_time = time.time()


#----------------------------------------------------------------------------#
# Read in a stack of uniform images
#----------------------------------------------------------------------------#
path = str(input('Please input the file path for the uniform images:    '))
dicom_stack = pydicom.dcmread(path)


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
slice_start = int(num_slices*0.4)
slice_end = int(num_slices*0.6)


#----------------------------------------------------------------------------#
# Define the number of slices to skip each time
# The higher the number, the quicker the analysis
#----------------------------------------------------------------------------#
slice_skip = 10

#----------------------------------------------------------------------------#
# Find the centre of the phantom
#----------------------------------------------------------------------------#
central_slice = stack[int(num_slices/2), :, :]


#----------------------------------------------------------------------------#
# Phantom diameter and pixel size used for initial guess of radius
#----------------------------------------------------------------------------#
phantom_diameter_mm = 80
pixel_size= 0.2
slice_thickness = 0.2
pix_area = pixel_size**2
vox_vol = pix_area*slice_thickness
phantom_diameter_pixels = int(phantom_diameter_mm/pixel_size) 


#----------------------------------------------------------------------------#
# Find outline of the phantom
#----------------------------------------------------------------------------#
edges = canny(central_slice, sigma=5, low_threshold=5, high_threshold=20)


#----------------------------------------------------------------------------#
# Set a range of likely radii for the phantom
#----------------------------------------------------------------------------#
radius = np.arange((phantom_diameter_pixels//2-10), (phantom_diameter_pixels//2)+10, 1)
hough_res = hough_circle(edges, radius)


#----------------------------------------------------------------------------#
# Find the actual radius and the centre of the phantom for each slice
#----------------------------------------------------------------------------#
accums, cx, cy, radii = hough_circle_peaks(hough_res, radius, total_num_peaks=1)



#----------------------------------------------------------------------------#

# Plot original image with edge of phantom overlaid
fig, ax = plt.subplots(ncols=1, nrows=1)
image = color.gray2rgb(central_slice)
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
    
    # Set RGB colour of the circle
    image[circy, circx] = (255, 0, 0)

ax.imshow(image)
ax.plot([cx], [cy], marker='o', markersize=1, color="red")
plt.show()
    
#----------------------------------------------------------------------------#



# START OF NPS ANALYSIS #

slice_used=[]
slice_nps_list =[]


# For each slice in the defined range
for slices in range(slice_start, slice_end):
    
    # Read in individual slices of the stack
    uniform_slice = stack[slices, :, :]
         
    #-------------------------------------------------------------------------#
    if slices % slice_skip == 0:
    
        # Track the slice being used
        slice_used = np.append(slice_used, [slices], axis=0)
    
        # Region of interest size in pixels
        roi_size = 64
        
        # Change in x,y coordinate to different ROIs
        delta_x = phantom_diameter_pixels/6
        delta_y = phantom_diameter_pixels/6
        
        #---------------------------------------------------------------------#     
        # We have four different ROIs.
        # In compass directions, NW, NE, SW, SE     
        #---------------------------------------------------------------------#         
        #---------------------------------------------------------------------#
             
        # For the NW quadrant
        cx_nw = cx-delta_x
        cy_nw = cy-delta_y
        
        start_x_nw = int(cx_nw - roi_size//2)
        end_x_nw = int(cx_nw + roi_size//2)
        start_y_nw = int(cy_nw - roi_size//2)
        end_y_nw = int(cy_nw + roi_size//2)
        
        # select the region of interest using NumPy indexing
        roi_nw = uniform_slice[start_y_nw:end_y_nw, start_x_nw:end_x_nw]
        
        #---------------------------------------------------------------------#   
        
        # For the NE quadrant
        cx_ne = cx+delta_x
        cy_ne = cy-delta_y
        
        start_x_ne = int(cx_ne - roi_size//2)
        end_x_ne = int(cx_ne + roi_size//2)
        start_y_ne = int(cy_ne - roi_size//2)
        end_y_ne = int(cy_ne + roi_size//2)
        
       # Select the region of interest using NumPy indexing
        roi_ne = uniform_slice[start_y_ne:end_y_ne, start_x_ne:end_x_ne]
     
       #-----------------------------------------------------------------------#
     
       # For the SW quadrant
        cx_sw = cx-delta_x
        cy_sw = cy+delta_y
        
        start_x_sw = int(cx_sw - roi_size//2)
        end_x_sw = int(cx_sw + roi_size//2)
        start_y_sw = int(cy_sw - roi_size//2)
        end_y_sw = int(cy_sw + roi_size//2)
        
        # Select the region of interest using NumPy indexing
        roi_sw = uniform_slice[start_y_sw:end_y_sw, start_x_sw:end_x_sw]

        #---------------------------------------------------------------------#

     
        # For the SE quadrant
        cx_se = cx+delta_x
        cy_se = cy+delta_y
        
        start_x_se = int(cx_se - roi_size//2)
        end_x_se = int(cx_se + roi_size//2)
        start_y_se = int(cy_se - roi_size//2)
        end_y_se = int(cy_se + roi_size//2)
        
        # select the region of interest using NumPy indexing
        roi_se = uniform_slice[start_y_se:end_y_se, start_x_se:end_x_se]
        
        #---------------------------------------------------------------------#
        #---------------------------------------------------------------------#
        
        
        
        #---------------------------------------------------------------------#
        # Put all the rois into one larger matrix
        #---------------------------------------------------------------------#     
        all_rois = [roi_nw, roi_ne, roi_sw, roi_se]
     
       
        # # Display the original image and an ROI using Matplotlib
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        # ax[0].imshow(roi_se, cmap='gray')
        # ax[0].set_title('South East ROI')
        # ax[1].imshow(uniform_slice, cmap='gray')
        # ax[1].add_patch(plt.Rectangle((start_x_nw, start_y_nw), roi_size, roi_size, linewidth=2, edgecolor='r', facecolor='none'))
        # ax[1].add_patch(plt.Rectangle((start_x_ne, start_y_ne), roi_size, roi_size, linewidth=2, edgecolor='g', facecolor='none'))
        # ax[1].add_patch(plt.Rectangle((start_x_sw, start_y_sw), roi_size, roi_size, linewidth=2, edgecolor='b', facecolor='none'))
        # ax[1].add_patch(plt.Rectangle((start_x_se, start_y_se), roi_size, roi_size, linewidth=2, edgecolor='k', facecolor='none'))
        # ax[1].set_title('ROIs overlaid on Uniform Slice')
        
        #---------------------------------------------------------------------#  


        #---------------------------------------------------------------------#
        # Prep for NPS measurement - spatial frequencies
        #---------------------------------------------------------------------# 
         
        # Define spatial frequencies
        mmInv = (1/pixel_size)/roi_size
        
        # Define spatial frequency ranges
        max_freq_range = mmInv*np.sqrt(2)*(roi_size//2)
        
        # Rearrange spatial frequencies so that 0 lp/mm corresponds to the centre
        lp_mm_vert_range = np.arange(-(roi_size//2),roi_size//2)*mmInv
        lp_mm_hori_range = np.arange(-(roi_size//2),roi_size//2)*mmInv
        
        # Create a meshgrid of spatial frequencies
        xu, yv = np.meshgrid(lp_mm_vert_range, lp_mm_hori_range, indexing='ij')
        
        # Re-bin spatial frequencies to the closest 0.05 lp/mm
        radDist = np.sqrt(xu**2 + yv**2)
        interval = 0.05
        
        average_over_bin = np.round(radDist//interval)
        binnedDistances = average_over_bin*interval
        binned_distances_maxVal = binnedDistances.max()
        
        # Define binned spatial frequency range
        freqRange = np.arange(0, binned_distances_maxVal, interval)
        lenFreq = len(freqRange)
        
       
        #---------------------------------------------------------------------#
        # Loop through each quadrant roi for each slice
        #---------------------------------------------------------------------#         
        
        nps_all_quad_in_slice = np.zeros(shape=(0,lenFreq))

        for quad in range(len(all_rois)):
            roi = all_rois[quad]
            
            # Fit 2D polynomial to data and detrend
            # The polynomial is subtracted from each ROI to remove large scale inhomogeneities
            mn = np.arange(roi.size) 
            def second_order_fit(mn, a, b, c, d, e, f):
                m = mn // roi.shape[0]
                n = mn % roi.shape[1]
                return a + b*m + c*n + d*m**2 + e*n**2 + f*m*n
            
            fitted_params, pcov = curve_fit(second_order_fit, mn, np.ravel(roi))
            fitted_im = second_order_fit(mn, *fitted_params)
            fitted_im_2d = fitted_im.reshape(roi.shape)
            
            detrended_image = (roi - fitted_im_2d)
            detrended_image = (detrended_image + np.mean(roi))
            roi  = detrended_image.astype(float)
            
            # # Display ROI for NPS analysis
            # plt.figure(2)
            # plt.title('ROI')
            # plt.imshow(roi, cmap='gray') 
            
            
            #-----------------------------------------------------------------#
            # Step 1: Take Fourier Transform of ROI
            #-----------------------------------------------------------------#     
            image_fourier_transform = np.fft.fft2(roi)
 
            #-----------------------------------------------------------------#
            # Step 2: Shift the zero-frequency to the centre
            #-----------------------------------------------------------------#  
            image_FT_centre_zero = np.fft.fftshift(image_fourier_transform)
        
            #-----------------------------------------------------------------#
            # Step 3: Square the Fourier Transform
            #-----------------------------------------------------------------#
            image_FT_squared = abs(image_FT_centre_zero)**2
              
            #-----------------------------------------------------------------#
            # Step 4: Calculate the NPS at each spatial frequency
            #-----------------------------------------------------------------#                                  
            power_spectrum = []
            
            for k in range(len(freqRange)):
                indices_location = np.where(binnedDistances == freqRange[k])
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    power_spectrum = np.append(power_spectrum, image_FT_squared[indices_location[0], indices_location[1]].mean())
            
            # This is the NPS for a quandrant in a slice
            nps_quad_in_slice = power_spectrum*((pixel_size**2)/(roi_size**2))
            nps_quad_in_slice[0] = 0

            # Append the NPS for each quandrant to 'nps'
            nps_all_quad_in_slice=np.append(nps_all_quad_in_slice,[nps_quad_in_slice], axis=0)        
            
            # This is the mean NPS for a given slice
            mean_nps_in_slice= nps_all_quad_in_slice.mean(axis=0)
                                   
            #-----------------------------------------------------------------#
            # Step 5: Plot the NPS for each quadrant of this slice
            #-----------------------------------------------------------------# 
            # plt.figure(1)
            # plt.plot(freqRange[1:len(freqRange)], nps_all_quad_in_slice[quad][1:len(freqRange)],'o')
            # plt.title('Noise Power Spectrum vs. Spatial Frequency')
            # plt.xlabel('Spatial Frequency (mm$^{-1}$)')
            # plt.ylabel('NPS (HU$^2$ mm$^2$)')
            # plt.show()     
        
        slice_nps_list.append(nps_all_quad_in_slice)
        
        # This is a 3d matrix:
        # Number of slices, nps of each quadrant, nps at a given spatial frequency
        all_nps = np.stack(slice_nps_list)  

# Get mean NPS values of each quadrant
mean_nps_quad = np.mean(all_nps, axis=0)

# Get mean NPS values of each slice
mean_nps_slice = np.mean(all_nps, axis=1)

# Get overall mean NPS values
mean_nps = np.mean(mean_nps_quad, axis=0)
filtered_mean_nps = sg(mean_nps,9,3)

#---------------------------------------------------------------------#
# Step 6: Plot the mean NPS for this region of the phantom
#---------------------------------------------------------------------#              
plt.figure(1)
plt.plot(freqRange, mean_nps_quad[0], 'ro', label = 'NW')
plt.plot(freqRange, mean_nps_quad[1], 'go', label = 'NE')
plt.plot(freqRange, mean_nps_quad[2], 'bo', label = 'SW')
plt.plot(freqRange, mean_nps_quad[3], 'ko', label = 'SE')
plt.title('Noise Power Spectrum vs. Spatial Frequency (Quadrants)')
plt.xlabel('Spatial Frequency (mm$^{-1}$)')
plt.ylabel('NPS (HU$^2$ mm$^2$)')
plt.legend(loc='best')


plt.figure(2)
for slices_used in range(len(mean_nps_slice)):
    plt.plot(freqRange, mean_nps_slice[slices_used][:], 'o')
    plt.title('Noise Power Spectrum vs. Spatial Frequency (Per Slice)')
    plt.xlabel('Spatial Frequency (mm$^{-1}$)')
    plt.ylabel('NPS (HU$^2$ mm$^2$)')

plt.figure(3)
plt.plot(freqRange, mean_nps, 'ro', freqRange,filtered_mean_nps)
plt.title('Noise Power Spectrum vs. Spatial Frequency (Overall)')
plt.xlabel('Spatial Frequency (mm$^{-1}$)')
plt.ylabel('NPS (HU$^2$ mm$^2$)')

#---------------------------------------------------------------------------# 
#---------------------------------------------------------------------------#

print("--- %s seconds ---" % round((time.time() - start_time),2))  




fig, ax = plt.subplots(ncols=1, nrows=1)
image = color.gray2rgb(central_slice)
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
    
    # Set RGB colour of the circle
    image[circy, circx] = (255, 0, 0)

ax.imshow(image)
ax.plot([cx], [cy], marker='o', markersize=1, color="red")
ax.plot([cx_se], [cy_se], marker='o', markersize=1, color="green")
ax.plot([cx_sw], [cy_sw], marker='o', markersize=1, color="yellow")
ax.plot([cx_ne], [cy_ne], marker='o', markersize=1, color="orange")
ax.plot([cx_nw], [cy_nw], marker='o', markersize=1, color="blue")
plt.show()
    