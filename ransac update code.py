import cv2
import numpy as np
import os
# read the input image
img_path = "E:\demonstration minato_ohashi_osaka_jpn_001.jpg"
if not os.path.exists(img_path):
    print(f"Image file not found: {img_path}")
    exit()

img = cv2.imread(img_path)

if img is None:
    print(f"Failed to load image: {img_path}")
    exit()

gaussian_img=cv2.GaussianBlur(img,(3,3),0,borderType=cv2.BORDER_CONSTANT)


# convert the input image to grayscale image
gray = cv2.cvtColor(gaussian_img,cv2.COLOR_BGR2GRAY)

sigma= 0.3
median=np.median(gray)
# median=np.median(img)
lower=int(max(0,(1.0-sigma)*median))
upper=int(min(255,(1.0+sigma)*median))
edges=cv2.Canny(gray,lower,upper)
minLineLength = 10
maxLineGap = 5

# apply probabilistic Hough transform
lines = cv2.HoughLinesP(edges,1,np.pi/180,minLineLength,maxLineGap)
for line in lines:
   for x1,y1,x2,y2 in line:
      cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
cv2.imshow("blur",gaussian_img)
cv2.imshow('houghlines.jpg',img)
cv2.imshow('edges', edges)
cv2.waitKey(0) 
cv2.imwrite('D:/ransac images/blur.png',gaussian_img)
cv2.imwrite('D:/ransac images/houghlines.jpg',img)
cv2.imwrite('D:/ransac images/canny edges.png',edges)


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
original_shape = img.shape
print(img.shape)

# Flatten Each channel of the Image
all_pixels  = img.reshape((-1,3))
print(all_pixels.shape)

dominant_colors = 10

km = KMeans(n_clusters=dominant_colors)
km.fit(all_pixels)
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
       n_clusters=4, n_init=10,
       random_state=None, tol=0.0001, verbose=0)
centers = km.cluster_centers_
print(centers) # In RGB Format
# Convert to Integer format
centers = np.array(centers,dtype='uint8')
print(centers)
i = 1

plt.figure(0,figsize=(8,2))

# Storing info in color array
colors = []

for each_col in centers:
    plt.subplot(1,dominant_colors,i)
    plt.axis("off")
    i+=1
    
    colors.append(each_col)
    
    # Color Swatch
    a = np.zeros((100,100,3),dtype='uint8')
    a[:,:,:] = each_col
    plt.imshow(a)
    
plt.show()

# Segmenting our original image
new_img = np.zeros((img.shape[0]*img.shape[1],3),dtype='uint8')
print(new_img.shape)
colors
km.labels_
# Iterate over the image
for ix in range(new_img.shape[0]):
    new_img[ix] = colors[km.labels_[ix]]

new_img = new_img.reshape((original_shape))    
plt.imshow(new_img)
plt.show()
if  os.path.isdir("D:/ransac images"):
    breakpoint
else:
     os.mkdir("D:/ransac images")
cv2.imwrite('D:/ransac images/image-ransac.png',new_img)

import numpy as np
from matplotlib import pyplot as plt
from typing import Union, Any, List, Optional, cast, Dict

from skimage.measure import LineModelND, ransac

from skimage import io
import math
import datetime

min_samples=3 #RANSAC parameter - The minimum number of data points to fit a model to.
#min_inliers_allowed=5 #Custom parameter  - A line is selected only if these many inliers are found

class RansacLineInfo(object):
    """Helper class to manage the information about the RANSAC line."""
    def __init__(self, inlier_points:np.ndarray, model:LineModelND):
        self.inliers=inlier_points #the inliers that were detected by RANSAC algo
        self.model=model    #The LinearModelND that was a result of RANSAC algo

    @property
    def unitvector(self):
        """The unitvector of the model. This is an array of 2 elements (x,y)"""
        return self.model.params[1]

def read_black_pixels(imagefilename:str):
    #returns a numpy array with shape (N,2) N points, x=[0], y=[1]
    #The coordinate system is Cartesian
    np_image=io.imread(imagefilename,as_gray=True)
    black_white_threshold=0
    if (np_image.dtype == 'float'):
        black_white_threshold=0.5
    elif (np_image.dtype == 'uint8'):
        black_white_threshold=128
    else:
        raise Exception("Invalid dtype %s " % (np_image.dtype))
    indices=np.where(np_image <= black_white_threshold)
    width=np_image.shape[1]
    height=np_image.shape[0]
    cartesian_y=height-indices[0]-1
    np_data_points=np.column_stack((indices[1],cartesian_y)) 
    return np_data_points, width,height

def extract_first_ransac_line(data_points:[], max_distance:int):
    """
    Accepts a numpy array with shape N,2  N points, with coordinates x=[0],y=[1]
    Returns 
         A numpy array with shape (N,2), these are the inliers of the just discovered ransac line
         All data points with the inliers removed
         The model line
    """
    
    
    model_robust, inliers = ransac(data_points, LineModelND, min_samples=min_samples,
                                   residual_threshold=max_distance, max_trials=1000)
    results_inliers=[]
    results_inliers_removed=[]
    for i in range(0,len(data_points)):
        if (inliers[i] == False):
            #Not an inlier
            results_inliers_removed.append(data_points[i])
            continue
        x=data_points[i][0]
        y=data_points[i][1]
        results_inliers.append((x,y))
    return np.array(results_inliers), np.array(results_inliers_removed),model_robust

def generate_plottable_points_along_line(model:LineModelND, xmin:int,xmax:int, ymin:int, ymax:int):
    """
    Computes points along the specified line model
    The visual range is 
    between xmin and xmax along X axis
        and
    between ymin and ymax along Y axis
    return shape is [[x1,y1],[x2,y2]]
    """
    unit_vector=model.params[1]
    slope=abs(unit_vector[1]/unit_vector[0])
    x_values=None
    y_values=None
    if (slope > 1):
        y_values=np.arange(ymin, ymax,1)
        x_values=model.predict_x(y_values)
    else:        
        x_values=np.arange(xmin, xmax,1)
        y_values=model.predict_y(x_values)

    np_data_points=np.column_stack((x_values,y_values)) 
    return np_data_points

def superimpose_all_inliers(ransac_lines,width:float, height:float):
    #Create an RGB image array with dimension heightXwidth
    #Draw the points with various colours
    #return the array

    new_image=np.full([height,width,3],255,dtype='int')
    colors=[(0,255,0),(255,255,0),(0,0,255)]
    for line_index in range(0,len(ransac_lines)):
        color=colors[line_index % len(colors)]
        ransac_lineinfo:RansacLineInfo=ransac_lines[line_index]
        inliers=ransac_lineinfo.inliers 
        y_min=inliers[:,1].min()
        y_max=inliers[:,1].max()
        x_min=inliers[:,0].min()
        x_max=inliers[:,0].max()
        plottable_points=generate_plottable_points_along_line(ransac_lineinfo.model, xmin=x_min,xmax=x_max, ymin=y_min,ymax=y_max)
        for point in plottable_points:
            x=int(round(point[0]))
            if (x >= width) or (x < 0):
                continue
            y=int(round(point[1]))
            if (y >= height) or (y < 0):
                continue
            new_y=height-y-1
            new_image[new_y][x][0]=color[0]
            new_image[new_y][x][1]=color[1]
            new_image[new_y][x][2]=color[2]
    return new_image

def extract_multiple_lines_and_save(inputfilename:str,iterations:int, max_distance:int,min_inliers_allowed:int):
    """
    min_inliers_allowed - a line is selected only if it has more than this inliers. The search process is halted when this condition is met
    max_distance - This is the RANSAC threshold distance from a line for a point to be classified as inlier
    """
    print("---------------------------------------")
    print("Processing: %s" % (inputfilename))
    folder_script=os.path.dirname(__file__)
    absolute_path=os.path.join(folder_script,"images/",inputfilename)

    results:List[RansacLineInfo]=[]
    all_black_points,width,height=read_black_pixels(absolute_path)
    print("Found %d pixels in the file %s" % (len(all_black_points),inputfilename))
    starting_points=all_black_points
    for index in range(0,iterations):
        if (len(starting_points) <= min_samples):
            print("No more points available. Terminating search for RANSAC")
            break
        inlier_points,inliers_removed_from_starting,model=extract_first_ransac_line(starting_points,max_distance=max_distance)
        if (len(inlier_points) < min_inliers_allowed):
            print("Not sufficeint inliers found %d , threshold=%d, therefore halting" % (len(inlier_points),min_inliers_allowed))
            break
        starting_points=inliers_removed_from_starting
        results.append(RansacLineInfo(inlier_points,model))
        print("Found %d RANSAC lines" % (len(results)))
    superimposed_image=superimpose_all_inliers(results,width,height)
    #Save the results
    io.imsave("D:\ransac images",superimposed_image)




# extract_multiple_lines_and_save("SmallCross.png",5)
# extract_multiple_lines_and_save("SmallCrossWithNoise.png",5)
# extract_multiple_lines_and_save("2ProminentLine.png",5)
#todo some problem in one of the above lines, excpetion

# extract_multiple_lines_and_save("1SmallLineWithNoise.png",iterations= 5,max_distance=1, min_inliers_allowed=15)
# extract_multiple_lines_and_save("2ProminentLineWithNoise.png",5,max_distance=3, min_inliers_allowed=5)
# extract_multiple_lines_and_save("3ProminentLineWithNoise.png",5, max_distance=3,min_inliers_allowed=5)
#extract_multiple_lines_and_save("WheelSpoke.png",iterations= 30,max_distance=1, min_inliers_allowed=50)


extract_multiple_lines_and_save('D:/ransac images/image-ransac.png',iterations= 30,max_distance=1, min_inliers_allowed=50)