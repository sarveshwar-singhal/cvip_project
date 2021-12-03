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
np.random.seed(2)
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random
# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random


def calc_ssd(des1, des2):
    left_ssd = np.zeros([des1.shape[0], des2.shape[0]])
    right_ssd = np.zeros([des2.shape[0], des1.shape[0]])
    for i in range(des1.shape[0]):
        for j in range(des2.shape[0]):
            ssd = ((des1[i]-des2[j]) ** 2).sum()
            ssd = np.sqrt(ssd)
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
    original_left = left_img.copy()
    original_right = right_img.copy()
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
    prev_count = 0
    final_set = set()
    for i in range(5000):
        key_index = np.random.randint(len(final_d2), size=4)
        M = np.empty([0,9])
        key_set = set()
        temp_set = set()
        count = 0
        for j in range(len(key_index)):
            key_pos = list(final_d2.keys())[key_index[j]]
            key_set.add(key_pos)
            x, y = kp2[key_pos].pt
            x1, y1 = kp1[final_d2[key_pos][0]].pt
            row = [[x, y, 1, 0, 0, 0, -x1*x, -x1*y, -x1],[0,0,0, x, y, 1, -y1*x, -y1*y, -y1]]
            row = np.array(row)
            M = np.append(M, row, axis=0)
        u,s,v = np.linalg.svd(M, full_matrices=False)
        h = v[-1].copy()
        h = h/h[-1]
        h = h.reshape([3,3])
        for key in final_d2.keys():
            if key in key_set:
                continue
            else:
                x, y = kp2[key].pt
                x1, y1 = kp1[final_d2[key][0]].pt
                right = np.array([x,y,1]).reshape([3,1])
                calculated = np.matmul(h, right)
                actual = np.array([x1, y1, 1]).reshape([3,1])
                ssd = ((calculated - actual)**2).sum()
                ssd = np.sqrt(ssd)
                if ssd <=5:
                    count += 1
                    temp_set.add(key)
        if count > prev_count:
            prev_count = count
            final_set = temp_set.copy()
    final_list = list(final_set)
    for j in range(len(final_list)):
        x, y = kp2[final_list[j]].pt
        x1, y1 = kp1[final_d2[final_list[j]][0]].pt
        row = [[x, y, 1, 0, 0, 0, -x1*x, -x1*y, -x1],[0,0,0, x, y, 1, -y1*x, -y1*y, -y1]]
        row = np.array(row)
        M = np.append(M, row, axis=0)
    u,s,v = np.linalg.svd(M, full_matrices=False)
    h = v[-1].copy()
    h = h/h[-1]
    h = h.reshape([3,3])
    corner = np.array([[0,0],[0,original_right.shape[1]],[original_right.shape[0],original_right.shape[1]],
              [original_right.shape[0], 0]]).astype(float).reshape(-1,1,2)
    offset = cv2.perspectiveTransform(corner, h).tolist()
    x = []
    y = []
    for i in offset:
        x.append(i[0][0])
        y.append(i[0][1])
    width = int(max(x) + min(x))
    height = int(max(y) + min(y))
    y_abs = int(min(y)).__abs__()
    temp_mat = np.array([[1,0,0],[0,1,y_abs+9],[0,0,1]])
    out = np.matmul(h, temp_mat)
    result_img = cv2.warpPerspective(original_right, out, (width, height))
    result_img[y_abs+11:y_abs+11+original_left.shape[0], 0:original_left.shape[1]] = original_left
    return result_img
    

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)


