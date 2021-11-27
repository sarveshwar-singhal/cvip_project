###############
##Design the function "findRotMat" to  return 
# 1) rotMat1: a 2D numpy array which indicates the rotation matrix from xyz to XYZ 
# 2) rotMat2: a 2D numpy array which indicates the rotation matrix from XYZ to xyz 
#It is ok to add other functions if you need
###############

import numpy as np
import cv2


#getting rotation matrix and then multiplying them
def findRotMat(alpha, beta, gamma):
    alpha = degToRadian(alpha)
    beta = degToRadian(beta)
    gamma = degToRadian(gamma)
    r1 = getMatrix('z', alpha)
    r2 = getMatrix('x', beta)
    r3 = getMatrix('z', gamma)
    rot1 = np.matmul(r3, np.matmul(r2,r1))
    r1 = getMatrix('z', -gamma)
    r2 = getMatrix('x', -beta)
    r3 = getMatrix('z', -alpha)
    # rot2 = np.matmul(np.matmul(r1,r2), r3)
    rot2 = np.matmul(r3, np.matmul(r2,r1))
    # print(rot1, rot2)
    return rot1, rot2


def degToRadian(angleInDegree):
    return (np.pi/180)*angleInDegree

# this function will return the rotation matrix 3*3 (2-d array) based on angle and axis
def getMatrix(axis, angle):
    arr = []
    if axis == 'x':
        arr = [(1,0,0), (0, np.cos(angle), -np.sin(angle)), (0, np.sin(angle), np.cos(angle))]
    elif axis == 'y':
        arr = [(np.cos(angle), 0, np.sin(angle)), (0, 1,0), (-np.sin(angle), 0, np.cos(angle))]
    else:
        arr = [(np.cos(angle), -np.sin(angle), 0), (np.sin(angle), np.cos(angle), 0), (0, 0, 1)]
    return np.array(arr)

if __name__ == "__main__":
    alpha = 45
    beta = 30
    gamma = 60
    rotMat1, rotMat2 = findRotMat(alpha, beta, gamma)
    print(rotMat1)
    print(rotMat2)
