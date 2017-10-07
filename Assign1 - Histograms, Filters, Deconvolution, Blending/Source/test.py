import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt




filepath = r"C:\Users\ajinkuriakose\Desktop\Computer Vision\Assignments\Assignment1\HW1-Filters\input3A.jpg"
filepath1 = r"C:\Users\ajinkuriakose\Desktop\Computer Vision\Assignments\Assignment1\HW1-Filters\input3B.jpg"
# img  = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
A = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
B = cv2.cvtColor(cv2.imread(filepath1), cv2.COLOR_BGR2RGB)
# cv2.imshow('raw_image', img)
# b, g, r = cv2.split(img)
def ft(img, newsize=None):
    # f = np.fft.fft2(img)
    f = np.fft.fft2(img, newsize)
    fshift = np.fft.fftshift(f)
    return fshift

def ift(shift):
    f_ishift = np.fft.ifftshift(shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

##Question 2
"""
def ft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift

def ift(shift):
    f_ishift = np.fft.ifftshift(shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

hpfshift = ft(img)
magnitude_spectrum = 20*np.log(np.abs(hpfshift))
rows, cols = img.shape
crow,ccol = rows/2 , cols/2
hpfshift[crow-10:crow+10, ccol-10:ccol+10] = 0
hp_img_back = ift(hpfshift)



lpfshift = ft(img)
# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols),np.uint8)
mask[crow-10:crow+10, ccol-10:ccol+10] = 1

# apply mask and inverse DFT
lpfshift = lpfshift*mask
lp_img_back = ift(lpfshift)

plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(hp_img_back, cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(lp_img_back, cmap = 'gray')
plt.title('Image after LPF'), plt.xticks([]), plt.yticks([])
plt.show()

"""


"""
##Question 3
gk = cv2.getGaussianKernel(21,5)
gk = gk * gk.T
imf = ft(img, (img.shape[0],img.shape[1]))
gkf = ft(gk, (img.shape[0],img.shape[1])) # so we can multiple easily
imconvf = imf/gkf
blurred = ift(imconvf)

"""

#Question 4


# generate Gaussian pyramid for A
A = A[:,:A.shape[0]]
B = B[:A.shape[0],:A.shape[0]]
G = A.copy()
gpA = [G]
for i in xrange(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in xrange(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in xrange(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in xrange(5,0,-1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in xrange(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])


cv2.imwrite('Pyramid_blending2.jpg',ls_)

k = cv2.waitKey(0)
cv2.destroyAllWindows()


