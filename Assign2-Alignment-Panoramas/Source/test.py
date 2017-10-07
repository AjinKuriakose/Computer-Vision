import os
import sys
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Perspective warping")
   print("2 Cylindrical warping")
   print("3 Bonus perspective warping")
   print("4 Bonus cylindrical warping")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image1] " + "[path to input image2] " + "[path to input image3] " + "[output directory]")

'''
Detect, extract and match features between img1 and img2.
Using SIFT as the detector/extractor, but this is inconsequential to the user.

Returns: (pts1, pts2), where ptsN are points on image N.
    The lists are "aligned", i.e. point i in pts1 matches with point i in pts2.

Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (pts1, pts2) = feature_matching(im1, im2)

    plt.subplot(121)
    plt.imshow(im1)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
    plt.subplot(122)
    plt.imshow(im2)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
'''
def feature_matching(img1, img2, savefig=False):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches2to1 = flann.knnMatch(des2,des1,k=2)

    matchesMask_ratio = [[0,0] for i in xrange(len(matches2to1))]
    match_dict = {}
    for i,(m,n) in enumerate(matches2to1):
        if m.distance < 0.7*n.distance:
            matchesMask_ratio[i]=[1,0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1,des2,k=2)
    matchesMask_ratio_recip = [[0,0] for i in xrange(len(recip_matches))]

    for i,(m,n) in enumerate(recip_matches):
        if m.distance < 0.7*n.distance: # ratio
            if m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx: #reciprocal
                good.append(m)
                matchesMask_ratio_recip[i]=[1,0]



    if savefig:
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask_ratio_recip,
                           flags = 0)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,recip_matches,None,**draw_params)

        plt.figure(),plt.xticks([]),plt.yticks([])
        plt.imshow(img3,)
        plt.savefig("feature_matching.png",bbox_inches='tight')

    return ([ kp1[m.queryIdx].pt for m in good ],[ kp2[m.trainIdx].pt for m in good ])


def getTransform(src, dst, method='affine'):
    pts1,pts2 = feature_matching(src,dst)

    src_pts = np.float32(pts1).reshape(-1,1,2)
    dst_pts = np.float32(pts2).reshape(-1,1,2)

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        # M = np.append(M, [[0,0,1]], axis=0)

    if method == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    return (M, pts1, pts2, mask)

def cylindricalWarpImage(img1, K, savefig=False):
    f = K[0,0]

    im_h,im_w = img1.shape

    # go inverse from cylindrical coord to the image
    # (this way there are no gaps)
    cyl = np.zeros_like(img1)
    cyl_mask = np.zeros_like(img1)
    cyl_h,cyl_w = cyl.shape
    x_c = float(cyl_w) / 2.0
    y_c = float(cyl_h) / 2.0
    for x_cyl in np.arange(0,cyl_w):
        for y_cyl in np.arange(0,cyl_h):
            theta = (x_cyl - x_c) / f
            h     = (y_cyl - y_c) / f

            X = np.array([math.sin(theta), h, math.cos(theta)])
            X = np.dot(K,X)
            x_im = X[0] / X[2]
            if x_im < 0 or x_im >= im_w:
                continue

            y_im = X[1] / X[2]
            if y_im < 0 or y_im >= im_h:
                continue

            cyl[int(y_cyl),int(x_cyl)] = img1[int(y_im),int(x_im)]
            cyl_mask[int(y_cyl),int(x_cyl)] = 255


    if savefig:
        plt.imshow(cyl, cmap='gray')
        plt.savefig("cyl.png",bbox_inches='tight')

    return (cyl,cyl_mask)

def RMSD(questionID, target, master):
    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or master_channel != target_channel):
        return -1
    else:
        nonZero_target = cv2.countNonZero(target)
        nonZero_master = cv2.countNonZero(master)

        if (questionID == 1):
           if (nonZero_target < 1200000):
               return -1
        elif(questionID == 2):
            if (nonZero_target < 700000):
                return -1
        else:
            return -1

        total_diff = 0.0;
        master_channels = cv2.split(master);
        target_channels = cv2.split(target);

        for i in range(0, len(master_channels), 1):
            dst = cv2.absdiff(master_channels[i], target_channels[i])
            dst = cv2.pow(dst, 2)
            mean = cv2.mean(dst)
            total_diff = total_diff + mean[0]**(1/2.0)

        return total_diff;

middle= cv2.imread(r"C:\Users\ajinkuriakose\Desktop\Computer Vision\Assignments\Assign3-Alignment-Panoramas\input1.png", 0)
right = cv2.imread(r"C:\Users\ajinkuriakose\Desktop\Computer Vision\Assignments\Assign3-Alignment-Panoramas\input2.png", 0)
left = cv2.imread(r"C:\Users\ajinkuriakose\Desktop\Computer Vision\Assignments\Assign3-Alignment-Panoramas\input3.png", 0)


# middle = cv2.copyMakeBorder(middle, 50, 50, 300, 300, cv2.BORDER_CONSTANT)
# h, w = middle.shape
# f = 400
# K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])  # mock calibration matrix
# middlecl,maskmiddle= cylindricalWarpImage(middle, K)
# h, w = left.shape
# f = 400
# K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])  # mock calibration matrix
# leftcl,maskleft= cylindricalWarpImage(left, K)
# h, w = right.shape
# f = 400
# K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])  # mock calibration matrix
# rightcl,maskright= cylindricalWarpImage(right, K)
#
#
#
# (M, pts1, pts2, mask) = getTransform(rightcl, middlecl)
# out = cv2.warpAffine(rightcl, M, (middlecl.shape[1], middlecl.shape[0]), dst=middlecl.copy(), borderMode=cv2.BORDER_TRANSPARENT)
# (M, pts1, pts2, mask) = getTransform(leftcl, out)
# outR = cv2.warpAffine(leftcl, M, (out.shape[1], out.shape[0]), dst=out.copy(), borderMode=cv2.BORDER_TRANSPARENT)
# maskR = cv2.warpAffine(maskleft, M, (out.shape[1], out.shape[0]), dst=maskmiddle.copy(), borderMode=cv2.BORDER_TRANSPARENT)
#
# new = outR + out*(1-maskR/255)
#
# plt.subplot(121),plt.imshow(outR, cmap='gray')
# plt.subplot(122),plt.imshow(new, cmap='gray')
# plt.show()

res = cv2.imread(r"C:\Users\ajinkuriakose\Desktop\Computer Vision\Assignments\Assign3-Alignment-Panoramas\example_output1.png", 0)

res1 = cv2.imread(r"C:\Users\ajinkuriakose\Desktop\Computer Vision\Assignments\Assign3-Alignment-Panoramas\Results\output_homography.png", 0)

print RMSD(1, res1, res)