# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output
   
# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

#calculates the cdf and returns the histogram equilized image.
def histEq(img_channel):
    hist, bins = np.histogram(img_channel.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img_channel_out = cdf[img_channel]

    return img_channel_out

def histogram_equalization(img_in):

   # Write histogram equalization here
   b, g, r = cv2.split(img_in)
   img_out = cv2.merge((histEq(b), histEq(g), histEq(r)))
   # img_out = img_in # Histogram equalization result
   # resnp = np.hstack((img_in, img_out))
   # cv2.imshow('resnp', resnp)
   return True, img_out
   
def Question1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   
   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)
   
   # Write out the result
   output_name = sys.argv[3] + "1.jpg"
   cv2.imwrite(output_name, output_image)

   return True
   
# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================
def ft(img,newsize=None):
    f = np.fft.fft2(img,newsize)
    fshift = np.fft.fftshift(f)
    return fshift

def ift(shift):
    f_ishift = np.fft.ifftshift(shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

def low_pass_filter(img_in):
   
   # Write low pass filter here
   rows, cols = img_in.shape
   crow, ccol = rows / 2, cols / 2
   lpfshift = ft(img_in)
   # create a mask first, center square is 1, remaining all zeros
   mask = np.zeros((rows, cols), np.uint8)
   mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 1

   # apply mask and inverse DFT
   lpfshift = lpfshift * mask
   img_out = ift(lpfshift)
   
   return True, img_out

def high_pass_filter(img_in):

   # Write high pass filter here
   hpfshift = ft(img_in)
   magnitude_spectrum = 20 * np.log(np.abs(hpfshift))
   rows, cols = img_in.shape
   crow, ccol = rows / 2, cols / 2
   hpfshift[crow - 10:crow + 10, ccol - 10:ccol + 10] = 0
   img_out = ift(hpfshift)
   # High pass filter result
   
   return True, img_out
   
def deconvolution(img):
   
   # Write deconvolution codes here
   gk = cv2.getGaussianKernel(21, 5)
   gk = gk * gk.T
   imf = ft(img, (img.shape[0], img.shape[1]))
   gkf = ft(gk, (img.shape[0], img.shape[1]))  # so we can multiple easily
   imconvf = imf / gkf
   img_out = ift(imconvf)
   img_out = np.array(img_out* 255, dtype=np.uint8)
   # img_out = img_in # Deconvolution result
   
   return True, img_out

def Question2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], 0);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);

   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)
   
   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)
   
   # Write out the result
   output_name1 = sys.argv[4] + "2.jpg"
   output_name2 = sys.argv[4] + "3.jpg"
   output_name3 = sys.argv[4] + "4.jpg"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, output_image3)
   
   return True
   
# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(A, B):

   # Write laplacian pyramid blending codes here
   # generate Gaussian pyramid for A
   A = A[:, :A.shape[0]]
   B = B[:A.shape[0], :A.shape[0]]
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
   for i in xrange(5, 0, -1):
      GE = cv2.pyrUp(gpA[i])
      L = cv2.subtract(gpA[i - 1], GE)
      lpA.append(L)

   # generate Laplacian Pyramid for B
   lpB = [gpB[5]]
   for i in xrange(5, 0, -1):
      GE = cv2.pyrUp(gpB[i])
      L = cv2.subtract(gpB[i - 1], GE)
      lpB.append(L)

   # Now add left and right halves of images in each level
   LS = []
   for la, lb in zip(lpA, lpB):
      rows, cols, dpt = la.shape
      ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
      LS.append(ls)

   # now reconstruct
   img_out = LS[0]
   for i in xrange(1, 6):
      img_out = cv2.pyrUp(img_out)
      img_out = cv2.add(img_out, LS[i])
   
   return True, img_out

def Question3():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
   
   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
   
   # Write out the result
   output_name = sys.argv[4] + "5.jpg"
   cv2.imwrite(output_name, output_image)
   
   return True

if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
      question_number = int(sys.argv[1])
     
      if (question_number == 1 and not(len(sys.argv) == 4)):
         help_message()
         sys.exit()
      if (question_number == 2 and not(len(sys.argv) == 5)):
         help_message()
         sys.exit()
      if (question_number == 3 and not(len(sys.argv) == 5)):
         help_message()
         sys.exit()
      if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
         print("Input parameters out of bound ...")
         sys.exit()

   function_launch = {
   1 : Question1,
   2 : Question2,
   3 : Question3,
   }

   # Call the function
   function_launch[question_number]()
