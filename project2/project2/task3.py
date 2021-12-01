"""
Morphology Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with commonly used morphology
binary image processing techniques. Use the proper combination of the four commonly used morphology operations, 
i.e. erosion, dilation, open and close, to remove noises and extract boundary of a binary image. 
Specifically, you are given a binary image with noises for your testing, which is named 'task3.png'.  
Note that different binary image might be used when grading your code. 

You are required to write programs to: 
(i) implement four commonly used morphology operations: erosion, dilation, open and close. 
    The stucturing element (SE) should be a 3x3 square of all 1's for all the operations.
(ii) remove noises in task3.png using proper combination of the above morphology operations. 
(iii) extract the boundaries of the objects in denoised binary image 
      using proper combination of the above morphology operations. 
Hint: 
• Zero-padding is needed before morphology operations. 

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy libraries, HOWEVER, 
you are NOT allowed to use any functions or APIs directly related to morphology operations.
Please implement erosion, dilation, open and close operations ON YOUR OWN.
"""

from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np


def morph_erode(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return erode_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology erosion on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """
    # TO DO: implement your solution here
    structuring_element = np.array([1,1,1,1,1,1,1,1,1], dtype='uint8').reshape([3,3])
    pad_img = np.pad(img, (1,1), 'constant', constant_values=(0,0))
    erode_img = np.ones(img.shape, dtype='int')
    for i in range(1,pad_img.shape[0]-1):
        for j in range(1,pad_img.shape[1]-1):
            mat = pad_img[i-1:i+2, j-1:j+2]
            if False in (structuring_element == mat):
                erode_img[i-1][j-1] = 0
    # imwrite('results/erode.jpg', erode_img)
    # exit(1)
    return erode_img


def morph_dilate(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return dilate_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology dilation on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """

    # TO DO: implement your solution here
    filter = np.array([1,1,1,1,1,1,1,1,1], dtype='uint8').reshape([3,3])
    pad_img = np.pad(img, (1,1), 'constant', constant_values=(0,0))
    dilate_img = np.zeros(img.shape, dtype='int')
    for i in range(1,pad_img.shape[0]-1):
        for j in range(1,pad_img.shape[1]-1):
            mat = pad_img[i-1:i+2, j-1:j+2]
            if True in (filter == mat):
                dilate_img[i-1][j-1] = 1
    return dilate_img


def morph_open(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return open_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology opening on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """

    # TO DO: implement your solution here
    erode_img = morph_erode(img)
    open_img = morph_dilate(erode_img)
    return open_img


def morph_close(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return close_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology closing on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """
    # TO DO: implement your solution here
    dilate_img = morph_dilate(img)
    close_img = morph_erode(dilate_img)
    return close_img


def denoise(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Remove noises from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """

    # TO DO: implement your solution here
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] == 255:
                img[i,j] = 1
    erode_img = morph_erode(img)
    open_img = morph_open(erode_img)
    dilate_img = morph_dilate(open_img)
    close_img = morph_close(dilate_img)
    for i in range(close_img.shape[0]):
        for j in range(close_img.shape[1]):
            if close_img[i,j] == 1:
                close_img[i,j] = 255
    denoise_img = close_img
    return denoise_img


def boundary(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Extract boundaries from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """

    # TO DO: implement your solution here
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] == 255:
                img[i,j] = 1
    erode_img = morph_erode(img)
    img = img-erode_img
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] == 1:
                img[i,j] = 255
    bound_img = img
    # raise NotImplementedError
    return bound_img


if __name__ == "__main__":
    img = imread('task3.png', IMREAD_GRAYSCALE)
    denoise_img = denoise(img)
    imwrite('results/task3_denoise.jpg', denoise_img)
    bound_img = boundary(denoise_img)
    imwrite('results/task3_boundary.jpg', bound_img)





