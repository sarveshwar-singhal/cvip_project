"""
 Grayscale Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with two commonly used 
image processing techniques: image denoising and edge detection. 
Specifically, you are given a grayscale image with salt-and-pepper noise, 
which is named 'task2.png' for your code testing. 
Note that different image might be used when grading your code. 

You are required to write programs to: 
(i) denoise the image using 3x3 median filter;
(ii) detect edges in the denoised image along both x and y directions using Sobel operators (provided in line 30-32).
(iii) design two 3x3 kernels and detect edges in the denoised image along both 45° and 135° diagonal directions.
Hint: 
• Zero-padding is needed before filtering or convolution. 
• Normalization is needed before saving edge images. You can normalize image using the following equation:
    normalized_img = 255 * frac{img - min(img)}{max(img) - min(img)}

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy for basic matrix calculations EXCEPT any function/operation related to convolution or correlation. 
You should NOT use any other libraries, which provide APIs for convolution/correlation ormedian filtering. 
Please write the convolution code ON YOUR OWN. 
"""

from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np

# Sobel operators are given here, do NOT modify them.
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(int)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(int)


def filter(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Apply 3x3 Median Filter and reduce salt-and-pepper noises in the input noise image
    """
    # TO DO: implement your solution here
    denoise_img = np.empty([0,img.shape[1]], dtype='uint8')
    # pad_top = np.array(np.zeros([1, img.shape[1]], dtype='uint8'))
    # pad_img = np.append(pad_top, img, axis=0)   #top padding
    # pad_img = np.append(pad_img, pad_top, axis=0)   #bottom padding
    # pad_left = np.array(np.zeros([pad_img.shape[0],1], dtype='uint8'))
    # pad_img = np.append(pad_left, pad_img, axis=1)  #left padding
    # pad_img = np.append(pad_img, pad_left, axis=1)  #right padding
    pad_img = np.pad(img, (1,1), 'constant', constant_values=(0,0))
    for i in range(1, pad_img.shape[0]-1):
        row = []
        for j in range(1, pad_img.shape[1]-1):
            med_filter = pad_img[i-1:i+2, j-1:j+2]
            row.append(np.median(med_filter))
        row = np.array(row, dtype='uint8')
        row = row.reshape(1,row.shape[0])
        denoise_img = np.append(denoise_img, row, axis=0)
    return denoise_img


def convolve2d(img, kernel):
    """
    :param img: numpy.ndarray, image
    :param kernel: numpy.ndarray, kernel
    :return conv_img: numpy.ndarray, image, same size as the input image

    Convolves a given image (or matrix) and a given kernel.
    """
    # TO DO: implement your solution here
    kernel = np.flip(kernel)
    pad_img = np.pad(img, (1,1), 'constant', constant_values=(0,0))
    conv_img = np.empty((0, pad_img.shape[1]-2), 'int')
    for i in range(1, pad_img.shape[0]-1):
        row = []
        for j in range(1, pad_img.shape[1]-1):
            inner_prod = 0
            sample = pad_img[i-1:i+2,j-1:j+2]
            inner_prod += np.inner(sample[0], kernel[0])
            inner_prod += np.inner(sample[1], kernel[1])
            inner_prod += np.inner(sample[2], kernel[2])
            row.append(inner_prod)
        row = np.array(row, dtype='int')
        row = row.reshape([1, row.shape[0]])
        conv_img = np.append(conv_img, row, axis=0)
    return conv_img


def norm_img(conv_img):
    """ This function will normalize the intensity b/w 0-255"""
    norm_conv_img = np.empty((0, conv_img.shape[1]), 'int')
    min = conv_img.min()
    mul = 255/ (conv_img.max() - min)
    for i in range(conv_img.shape[0]):
        row =[]
        for j in range(conv_img.shape[1]):
            norm = (conv_img[i,j] - min) * mul
            row.append(norm)
        row = np.array(row, dtype='int')
        row = row.reshape([1, row.shape[0]])
        norm_conv_img = np.append(norm_conv_img, row, axis=0)
    return norm_conv_img


def edge_detect(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_x: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_y: numpy.ndarray(int), image, same size as the input image, edges along y direction
    :return edge_mag: numpy.ndarray(int), image, same size as the input image, 
                      magnitude of edges by combining edges along two orthogonal directions.

    Detect edges using Sobel kernel along x and y directions.
    Please use the Sobel operators provided in line 30-32.
    Calculate magnitude of edges by combining edges along two orthogonal directions.
    All returned images should be normalized to [0, 255].
    """
    # TO DO: implement your solution here
    convolve_x = convolve2d(img, sobel_x)
    convolve_y = convolve2d(img, sobel_y)
    edge_x = norm_img(convolve_x)
    edge_y = norm_img(convolve_y)
    edge_mag = np.empty([0, convolve_x.shape[1]], 'float')
    for i in range(convolve_x.shape[0]):
        row = []
        for j in range(convolve_x.shape[1]):
            val = np.sqrt(convolve_x[i,j] ** 2 + convolve_y[i,j] ** 2)
            row.append(val)
        row = np.array(row, dtype='float')
        row = row.reshape([1, row.shape[0]])
        edge_mag = np.append(edge_mag, row, axis=0)
    edge_mag = norm_img(edge_mag)
    return edge_x, edge_y, edge_mag


def edge_diag(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_45: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_135: numpy.ndarray(int), image, same size as the input image, edges along y direction

    Design two 3x3 kernels to detect the diagonal edges of input image. Please print out the kernels you designed.
    Detect diagonal edges along 45° and 135° diagonal directions using the kernels you designed.
    All returned images should be normalized to [0, 255].
    """
    # TO DO: implement your solution here
    ker45 = np.array([[0,1,2],[-1,0,1],[-2,-1,0]])
    ker135 = np.fliplr(ker45)
    conv45 = convolve2d(img, ker45)
    conv135 = convolve2d(img, ker135)
    edge_45 = norm_img(conv45)
    edge_135 = norm_img(conv135)
    # raise NotImplementedError
    print(ker45) # print the two kernels you designed here
    print(ker135)
    return edge_45, edge_135


if __name__ == "__main__":
    # noise_img = imread('task2.png', IMREAD_GRAYSCALE)
    # denoise_img = filter(noise_img)
    # imwrite('results/task2_denoise.jpg', denoise_img)
    # exit(10)      #remove
    denoise_img = imread('results/task2_denoise.jpg', IMREAD_GRAYSCALE)     #remove
    edge_x_img, edge_y_img, edge_mag_img = edge_detect(denoise_img)
    imwrite('results/task2_edge_x.jpg', edge_x_img)
    imwrite('results/task2_edge_y.jpg', edge_y_img)
    imwrite('results/task2_edge_mag.jpg', edge_mag_img)
    # exit(10)    #remove
    edge_45_img, edge_135_img = edge_diag(denoise_img)
    imwrite('results/task2_edge_diag1.jpg', edge_45_img)
    imwrite('results/task2_edge_diag2.jpg', edge_135_img)





