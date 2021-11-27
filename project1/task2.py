###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates. 
#                                            True if the intrinsic parameters are invariable.
#It is ok to add other functions if you need
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners
# from cv2 import imshow, destroyAllWindows, waitKey, resize

def calibrate(imgname):
    img_obj = imread(imgname)
    img_cvt = cvtColor(img_obj, COLOR_BGR2GRAY)
    pattFound, corners = findChessboardCorners(img_cvt, ( 9,4), None)
    x = drawChessboardCorners(img_obj,(9,4), corners=corners, patternWasFound= pattFound)
    # temp = resize(x, (960,450))
    # imshow('img',temp)
    # waitKey(0)
    # destroyAllWindows()
    crit = (TERM_CRITERIA_EPS+TERM_CRITERIA_MAX_ITER, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER)
    sub = cornerSubPix(img_cvt, corners, (9,4), (0,0), crit)
    a=[]
    for cord in sub:
        a.append(cord.item([0][0]))
        a.append(cord.item([1][0]))
    # print(a)
    # exit(10)
    worldCC = np.array([[40,0,10,1],[30,0,10,1],[20,0,10,1],[10,0,10,1],[0,0,10,1], [0,10,10,1],[0,20,10,1],
                        [0,30,10,1],[0,40,10,1],
               [40,0,20,1],[30,0,20,1],[20,0,20,1],[10,0,20,1],[0,0,20,1], [0,10,20,1],[0,20,20,1],
                        [0,30,20,1],[0,40,20,1],
               [40,0,30,1],[30,0,30,1],[20,0,30,1],[10,0,30,1],[0,0,30,1], [0,10,30,1],[0,20,30,1],[0,30,30,1],
                        [0,40,30,1],
               [40,0,40,1],[30,0,40,1],[20,0,40,1],[10,0,40,1],[0,0,40,1], [0,10,40,1],[0,20,40,1],
                        [0,30,40,1],[0,40,40,1]])
    # worldCC = np.array(worldCC)
    # print(type(worldCC))
    # print(worldCC.shape)
    temp = []
    temp1 = np.empty((0,4), float)
    temp2 = np.empty((0,4), float)
    temp3 = np.empty((0,4), float)
    for wor in worldCC:
        # print(wor, wor.shape, type(wor))
        # print(wor.reshape(3,1))
        wor = wor.reshape(1,4)
        # print(wor, wor.shape, type(wor))
        # exit(10)
        # np.concatenate((temp1, wor))
        # print(temp1)
        temp1= np.concatenate([temp1,wor], axis=0)
        temp1 = np.concatenate([temp1, [[0,0,0,0]]], axis=0)
        # print(temp1)
        temp2 = np.concatenate([temp2, [[0,0,0,0]]], axis=0)
        temp2= np.concatenate([temp2,wor], axis=0)
        # print(temp2)
        temp3= np.concatenate([temp3,wor], axis=0)
        temp3= np.concatenate([temp3,wor], axis=0)
        # print(temp3)
        # exit(10)
    # temp1 =np.matrix(np.array(temp1))

    # temp3 = np.array(temp3)
    for i in range(0,72):
        temp3[i] = temp3[i]*a[i]*(-1)

    #projection matrix temp
    temp = np.concatenate((np.concatenate((temp1,temp2), axis=1), temp3), axis=1)
    u,s,v = np.linalg.svd(temp)
    # print(v[-1])
    m = v[-1].reshape(3,4)
    # print(m)
    x = m[2][:3]
    xt = x.transpose()
    lam = 1/np.sqrt(np.matmul(x, xt))   #lambda value
    # print(lam)
    m = m*lam
    m1 = m[0][:3]
    m2 = m[1][:3]
    m3 = m[2][:3]
    ox = np.matmul(m1.T, m3)
    oy = np.matmul(m2.T, m3)
    fx = np.sqrt((np.matmul(m1.T, m1))-ox)
    fy = np.sqrt((np.matmul(m2.T, m2))-oy)
    return  [fx, fy, ox, oy], False
    # print(pattFound, corners)

    # print(img)

    # findChessboardCorners(imgname, patternSize=True)
    #......
    

if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)