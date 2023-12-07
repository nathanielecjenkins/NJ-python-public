# Library imports
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import time

def rescale(image, scale_percent):
    print('Original Dimensions : ',image.shape)
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
      
    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
     
    print('Resized Dimensions : ',image.shape)
    return image

def cluster(image,k):
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    print('clustering...')
    print('Original Dimensions : ',image.shape)
    pixel_vals = image.reshape((-1,3))
    # Convert to float type
    pixel_vals = np.float32(pixel_vals)
    #the below line of code defines the criteria for the algorithm to stop running, 
    #which will happen is 100 iterations are run or the epsilon (which is the required accuracy) 
    #becomes 85%
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
     
    # then perform k-means clustering with number of clusters defined as 3
    #also random centres are initially choosed for k-means clustering
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    #centers = Kx3
    #Reshape
    segmented_data = centers[labels.flatten()] 
    segmented_image = segmented_data.reshape((image.shape))
    print(labels.shape,'labels')    
    
    print(centers.shape,'centers')
    
    print(segmented_image.shape,'seg_data')
    # convert data into 8-bit values
    centers = np.uint8(centers)
    
     
    # reshape data into the original image dimensions
    
    return(segmented_image)

# Import image
image = cv2.imread("C:\\Users\\Noodleman\\Desktop\\temp.jpg")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Resize image
image = rescale(image, 5)

#Iterative cluster/blur
for x in range(0, 2):
    #Blur image
    prev_img = image
    blurred = cv2.GaussianBlur(image,(31,31),0)
    
    #Cluster image
    image = cluster(blurred,4)
    
    #Perform edge detection
    #canny = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),100,200)
    canny = image
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 4, 1)
    plt.imshow(prev_img)
    plt.axis('off')
    plt.title("previous") 
    fig.add_subplot(1, 4, 2)
    plt.imshow(blurred)
    plt.axis('off')
    plt.title("blurred") 
    fig.add_subplot(1, 4, 3)
    plt.imshow(image)
    plt.axis('off')
    plt.title("updated/clustered") 
    fig.add_subplot(1, 4, 4)
    plt.imshow(canny) 
    plt.axis('off') 
    plt.title("edge detection") 
    plt.show()
    time.sleep(3)

#Save results
directory = 'C:\\Users\\Noodleman\\Desktop'
os.chdir(directory)
print("Before saving image:")   
print(os.listdir(directory))
filename = 'Updated_Image.jpg'
  
# Using cv2.imwrite() method 
# Saving the image 
cv2.imwrite(filename, image) 
  
# List files and directories in CWD 
print("After saving image:")   
print(os.listdir(directory)) 


