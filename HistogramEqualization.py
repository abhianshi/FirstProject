from PIL import Image
import numpy as np
from pylab import *
from scipy import *
import matplotlib.pyplot as plt


# Creating the histogram of a given image by calculating the count of pixel values ranging from 0 to 255
def histogram(I):
    H = np.zeros(L)
    for i in range(row):
        for j in range(column):
            H[int(I[i][j])] += 1
    return H

# Normalized Probability
def normprob(H):
    p = np.zeros(L)
    p[0] = H[0]
    for i in range(1,L):
        p[i] = p[i-1] + H[i]
    for i in range(L):
        p[i] = p[i]/(row*column)
    return p

# Implementing the Histogram Equalization using cdf-cumulative distribution function and the number of pixels in the image
# Slightly different from the method discussed in class
def histogramEqualization(I,H):
    h = np.zeros(L)
    out = np.zeros(I.shape)
    cdf = np.zeros(L)
    cdf[0] = H[0]
    for i in range(1,L):
        cdf[i] = cdf[i-1]+H[i]
        h[i] = round(((cdf[i] - cdf[0])/(n-1))*(L-1))

    # Create the output image using histogram equalization values
    for i in range(row):
        for j in range(column):
            out[i][j] = h[I[i][j]]

    return out


# Implementing the Histogram Equalization using normalized probability and the number of pixels in the image
# Method discussed in the class
def histogramEqualization2(I,p):
    out = np.zeros(I.shape)

    # Create the output image using histogram equalization values
    for i in range(row):
        for j in range(column):
            #print(floor((L-1) * p[I[i][j]]))
            out[i][j] = int(floor((L-1) * p[I[i][j]]))
    #print(out)
    return out


alpha = 50
b = 150
beta = 2
# Clipping image - Basically this method divides image into three parts with given values

# Pixels whose value is less than alpha(50) is set to 0 (all black)
# Pixels whose value lies between alpha and b is first subtracted by alpha and then multiplied by beta
# In this case, pixel 51 becomes 2, 60 becomes 20, 140 becomes 180 and so on
# Pixels whose value is greater than b, have constant value 200 in this case
# So, pixels 0-50 becomes 0 and pixels 150-255 becomes 200 and pixels 50-150 are evenly distributed in the range 2 - 198

# After Clipping, extreme white and extreme black part of the image is assigned unique values
# Clipping occurs when an image is over-exposured and the image boundries between a certain range of pixels is all set to a fixed pixel value
# If we compare the histogram before and after clipping, we can see that the Probability of pixels whose value is between alpha and b is decreased.
# Number of white pixels increase in most cases after Clipping
def clipping(I,alpha, b, beta):
    global clip
    clip = np.zeros(I.shape)
    for i in range(row):
        for j in range(column):
            if(I[i][j] < alpha):
                clip[i][j] = 0
            elif(I[i][j]>= alpha and I[i][j] < b):
                clip[i][j] = beta*(I[i][j]-alpha)
            else:
                clip[i][j] = beta*(b-alpha)
    return clip


# Range compression - It compresses the image values with logarithmic function
# Log10 values of Grey-scale image(0-255) lies between 0 and 2.40823996531
# So, if value of c is greater than 1000, then almost all the range compressed values will go beyond 255 and a white image is formed
# With c = 110100, we will mostly get a full white image with only very few black dots, whose pixel value is 0.
# If some pixel value is 0, c won't have any effect on that pixel. C can be as large as possible.
# Very small value of c means that we will get very dark image
# Very high value of c means that we will get very bright image
def rangeCompression(I,c):
    global rangeComp
    rangeComp = np.zeros(I.shape)
    for i in range(row):
        for j in range(column):
            if(c*(log10(1+I[i][j])) > 255):
                rangeComp[i][j] = 255
            else:
                rangeComp[i][j] = c*(log10(1+I[i][j]))
    return rangeComp


#Read grey scale image and converting it into grey scale using convert function as some images have 3 values in shape function
# First two values are the rows and columns of the image and the third value is the channel, so to remove the third channel,convert is used
img = Image.open("/Users/abhianshusingla/Downloads/Man.jpg").convert('L')

# Store input image in matrix I
I = array(img)
row, column = I.shape

# Size of array
n = row*column

# Grey-scale pixel values
L = 256

# Calling histogram and histogram Equalization functions
H = histogram(I)

#Both the Histogram Equalized methods give approximately same result
#out = histogramEqualization(I,H)
p = normprob(H)
out = histogramEqualization2(I,p)

# Clipping
clip = clipping(I,alpha,b,beta)
#plt.hist(clip.transpose(), bins = 5, normed=1, facecolor='r')

# Range Compression
c = 1000
range1  = rangeCompression(I,c)
#plt.hist(range2.transpose(), bins = 5, normed=1, facecolor='r')

c = 110100
range2 = rangeCompression(I,c)
#plt.hist(range2.transpose(), bins = 5, normed=1, facecolor='r')

c = 100
range3 = rangeCompression(I,c)
c = 200
range4 = rangeCompression(I,c)

# Showing the image results
ax = plt.subplot(4,2,1)
ax.set_title('Image')
ax.imshow(img,cmap = 'gray')
# Plotting the histogram of Image I
plt.subplot(4,2,2)
plt.hist(I.transpose(), bins = 51,normed=1, facecolor='b')

ax = plt.subplot(4,2,3)
ax.set_title('Histogram Equalized Image')
ax.imshow(out,cmap = 'gray')
# Plotting the histogram of Image out
# Transposing the image to remove the UserWarning: 2D hist input should be nsamples x nvariables; this looks transposed
plt.subplot(4,2,4)
plt.hist(out.transpose(), bins = 51, normed=1, facecolor='g')

# Clipped Image
ax = plt.subplot(4,2,5)
ax.set_title('Clipped')
ax.imshow(clip,cmap = 'gray')

# Range Compressed Image
ax = plt.subplot(4,2,6)
ax.set_title('c = 100')
ax.imshow(range3,cmap = 'gray')
ax = plt.subplot(4,2,7)
ax.set_title('c = 1000')
ax.imshow(range1,cmap = 'gray')
ax = plt.subplot(4,2,8)
ax.set_title('c = 110100')
ax.imshow(range2,cmap = 'gray')

plt.show()
