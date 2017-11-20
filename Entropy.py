from PIL import Image
import numpy as np
from matplotlib.pylab import *
from scipy import *



# Creating the histogram of a given image by calculating the count of pixel values ranging from 0 to 255
def histogram(I):
    h = np.zeros(L)
    for i in range(row):
        for j in range(column):
            h[I[i][j]] += 1
    return h

# Calculating the probability of a particular pixel in an image using histogram values and total pixel values
def probability(h):
    p = np.zeros(L)
    for i in range(L):
        p[i] = h[i]/n
    return p

# Class correspond to background of the image using the pdf and entropy formula
def classA(t,P,p):
    E = 0
    if(P == 0):
        return E
    for i in range(t):
        # Check to avoid the zero value inside logarithmic function
        if(p[i] != 0):
            const = p[i]/P
            E += (-1)*const*log(const)
    return E

# Class correspond to foreground of the image using the pdf and entropy formula
def classB(t,P,p):
    E = 0
    if((1-P) == 0):
        return E
    for i in range(t, L-1):
        # Check to avoid the zero and negative values inside logarithmic function as for very few images, negative values are coming in logarithmic function and complex numbers are produced
        if(p[i] != 0 and (p[i]/(1-P)) > 0):
            const = p[i]/(1-P)
            E += (-1)*const*log(const)
    return E

# Calculate the total entropy, i.e., summation of Class A entropy and Class B entropy
def totalEntropy(p):
    global TE
    TE = np.zeros(L)
    P = 0
    for t in range(L):
        P += p[t]
        TE[t] = (classA(t,P,p) + classB(t,P,p))

# Finding the value of T for which entropy is maximum
def maximumEntropy():
    max = - float("inf")
    index = 0
    for i in range(L-1):
        if(TE[i] > max):
            max = TE[i]
            index = i
    return index

# Applying threshold to create binary images
# value 0 means black and value 255 means white
# Here, 255 is considered as 1 as is done for voltages in binary numbers
def binaryImage(T):
    bo = np.zeros(I.shape)
    for i in range(row):
        for j in range(column):
            if(I[i][j] >= T):
                bo[i][j] = 255
    return bo


#Read grey scale image and store it in matrix I
I = array(Image.open("/Users/abhianshusingla/Downloads/Boat.jpg"))

row, column = I.shape
n = row*column
L = 256

# Calling histogram and probability distribution functions
hist = histogram(I)
prob = probability(hist)

# Calling totalEntropy function and creating binary image output
totalEntropy(prob)
T = maximumEntropy()
O = binaryImage(T)

# Foreground images are shown
# Similary, Background image can be shown by inverting the values 0 and 255

# Plotting the original image and the output binary image
plt.suptitle('Entropy')
ax = plt.subplot(1,2,1)
ax.set_title('Original Image')
ax.imshow(I,cmap = 'gray')
ax = plt.subplot(1,2,2)
ax.set_title('Binary Image')
ax.imshow(O,cmap = 'gray')
plt.show()
