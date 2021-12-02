"""
Image Stitching Problem
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to stitch two images of overlap into one image.
You are given 'left.jpg' and 'right.jpg' for your image stitching code testing. 
Note that different left/right images might be used when grading your code. 

To this end, you need to find keypoints (points of interest) in the given left and right images.
Then, use proper feature descriptors to extract features for these keypoints. 
Next, you should match the keypoints in both images using the feature distance via KNN (k=2); 
cross-checking and ratio test might be helpful for feature matching. 
After this, you need to implement RANSAC algorithm to estimate homography matrix. 
(If you want to make your result reproducible, you can try and set fixed random seed)
At last, you can make a panorama, warp one image and stitch it to another one using the homography transform.
Note that your final panorama image should NOT be cropped or missing any region of left/right image. 

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
If you intend to use SIFT feature, make sure your OpenCV version is 3.4.2.17, see project2.pdf for details.
"""

import cv2
import numpy as np
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random
# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random


def calc_ssd(des1, des2):
    left_ssd = np.zeros([des1.shape[0], des2.shape[0]])
    right_ssd = np.zeros([des2.shape[0], des1.shape[0]])
    for i in range(des1.shape[0]):
        for j in range(des2.shape[0]):
            ssd = ((des1[i]-des2[j]) ** 2).sum()
            left_ssd[i,j] = ssd
            right_ssd[j,i] = ssd
    return left_ssd, right_ssd


def find_nearest(left_ssd, right_ssd):
    left_d1 = {}
    right_d1 = {}
    for i in range(left_ssd.shape[0]):
        row = left_ssd[i]
        min1 = row.min()
        min1_ind = row.argmin()
        row[min1_ind] = np.infty
        min2 = row.min()
        min2_ind = row.argmin()
        row[min1_ind] = min1
        left_d1[i] = [min1_ind, min2_ind, min1, min2]
    for i in range(right_ssd.shape[0]):
        row = right_ssd[i]
        min1 = row.min()
        min1_ind = row.argmin()
        row[min1_ind] = np.infty
        min2 = row.min()
        min2_ind = row.argmin()
        row[min1_ind] = min1
        right_d1[i] = [min1_ind, min2_ind, min1, min2]
    return left_d1, right_d1


def potential_match(left_d1, right_d1):    #dictionary containing nearest matches in this case 2
    potential_match = {}
    for key in left_d1.keys():
        if right_d1[left_d1[key][0]][0] == key and right_d1[left_d1[key][1]][1] == key:
            potential_match[key] = left_d1[key]
    return potential_match


def final_match(potential_d):   #potential_d: dictionary contains valid match
    final_d = {}
    for key in potential_d.keys():
        val = potential_d[key]
        if val[2]/val[3] < 0.7:
            final_d[key] = potential_d[key]
    return final_d


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """
    # TO DO: implement your solution here
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(left_img, None)
    kp2, des2 = sift.detectAndCompute(right_img, None)
    left_ssd, right_ssd = calc_ssd(des1, des2)
    left_d1, right_d1 = find_nearest(left_ssd, right_ssd)
    potential_d1 = potential_match(left_d1, right_d1)
    potential_d2 = potential_match(right_d1, left_d1)
    final_d1 = final_match(potential_d1)
    final_d2 = final_match(potential_d2)
    print(final_d1)
    print(final_d2)
    exit(10)
    return result_img
    

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)


