import cv2
import numpy as np
from numpy.fft import fft2

# get abs from complex matrix
def absMatrix(matrix):
    return np.sqrt(np.square(matrix.real) + np.square(matrix.imag))

# get 2D Gaussian value    
def gaussian2D(x , y, sigma, ux, uy):
    gaussian = np.exp(-((x-ux)*(x-ux)+(y-uy)*(y-uy))/(2*sigma*sigma))
    return gaussian/(2*np.pi*sigma*sigma)

# get Gaussian value filter kernel
def gaussianKernel(kernel_size, sigma=1):
    h_size = kernel_size//2
    gaussian_kernel = np.zeros((kernel_size,kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            gaussian_kernel[i,j] = gaussian2D(i , j, sigma, h_size, h_size)
    gaussian_kernel = gaussian_kernel/np.sum(gaussian_kernel)
    return gaussian_kernel

# get padding image use the border value
def samePadding(image, n):
    row,col = image.shape
    padding_image = np.zeros((row+n+n,col+n+n))
    padding_image[n:-n,n:-n] = image
    for i in range(n):
        padding_image[i,n:-n] = image[0,:]
        padding_image[-i-1:,n:-n] = image[-1,:]
        padding_image[n:-n,i] = image[:,0]
        padding_image[n:-n,-i-1] = image[:,-1]
    padding_image[:n,:n] = image[0,0]
    padding_image[:n,-n:] = image[0,-1]
    padding_image[-n:,:n] = image[-1,0]
    padding_image[-n:,-n:] = image[-1,-1]
    return padding_image

# get padding image use zero value
def zeroPadding(image, n):
    row,col = image.shape
    padding_image = np.zeros((row+n+n,col+n+n))
    padding_image[n:-n,n:-n] = image
    return padding_image

# get padding image use the border value with RGB channel
def paddingRGB(image, n, pad="same"):
    B_image = image[:,:,0]
    G_image = image[:,:,1]
    R_image = image[:,:,2]
    pad_B_image = samePadding(B_image, n)
    pad_G_image = samePadding(G_image, n)
    pad_R_image = samePadding(R_image, n)
    padding_image = np.dstack((pad_B_image, pad_G_image, pad_R_image))
    return padding_image

# remove padding border from image
def dePadding(padding_image, n):
    image = padding_image[n:-n,n:-n]
    return image

# 2D convolution operation
def conv2D(image, kernel):
    new_image = np.array(image)
    k_size = kernel.shape[0]
    padding_image = samePadding(image, k_size//2)
    row,col = image.shape
    for i in range(row):
        for j in range(col):
            sub_region = padding_image[i:i+k_size,j:j+k_size]
            new_image[i,j] = np.sum(np.multiply(sub_region, kernel))
    return new_image

# get half down sampling image
def downSamplingHalf(image):
    if image.ndim == 3:
        return image[::2, ::2, :]
    return image[::2, ::2]

# Gaussian pyramid down to next level
def pyramidDown(image):
    gaussian_kernel = gaussianKernel(5, 1)
    smooth_image = conv2D(image, gaussian_kernel)
    sub_sample_image = downSamplingHalf(smooth_image)
    return sub_sample_image

# Gaussian pyramid down to next level with RGB channel
def pyramidDownRGB(image):
    gaussian_kernel = gaussianKernel(5, 1)
    B_image = image[:,:,0]
    G_image = image[:,:,1]
    R_image = image[:,:,2]
    smooth_B_image = conv2D(B_image, gaussian_kernel)
    smooth_G_image = conv2D(G_image, gaussian_kernel)
    smooth_R_image = conv2D(R_image, gaussian_kernel)
    smooth_image = np.dstack((smooth_B_image, smooth_G_image, R_image))
    sub_sample_image = downSamplingHalf(smooth_image)
    return sub_sample_image

# get image log spectrum
def logSpectrum(image):
    phase_shift_image = np.array(image)
    phase_shift_image[::2, ::2] = -phase_shift_image[::2, ::2]
    phase_shift_image[1::2, 1::2] = -phase_shift_image[1::2, 1::2]
    freq_domain_image = fft2(phase_shift_image)
    spectrum = absMatrix(freq_domain_image)
    spectrum = np.log(spectrum+1)
    spectrum = (spectrum-np.min(spectrum))/(np.max(spectrum)-np.min(spectrum))
    return spectrum

# resize image with nearest neighbor interpolation
def resize(image, to_row, to_col):
    if(image.ndim == 3):
        row, col, channel = image.shape
        h_scale = row/to_row
        w_scale = col/to_col
        newImage = np.zeros((to_row,to_col, channel))

        row_index, col_index = np.indices((to_row, to_col))
        map_row = np.rint((row_index.astype(float)+1)*h_scale-1).astype(int)
        map_col = np.rint((col_index.astype(float)+1)*w_scale-1).astype(int)
        newImage[row_index,col_index,:] = image[map_row,map_col,:]
        return newImage
        
    else:
        row, col = image.shape
        h_scale = row/to_row
        w_scale = col/to_col
        newImage = np.zeros((to_row,to_col))

        ## slow
        #for i in range(to_row):
        #    for j in range(to_col):
        #        r = int(np.round((i+1)*h_scale-1))
        #        c = int(np.round((j+1)*w_scale-1))
        #        newImage[i,j] = image[r,c]

        ## fast       
        row_index, col_index = np.indices((to_row, to_col))
        map_row = np.rint((row_index.astype(float)+1)*h_scale-1).astype(int)
        map_col = np.rint((col_index.astype(float)+1)*w_scale-1).astype(int)
        newImage[row_index,col_index] = image[map_row,map_col]
        return newImage

# save float matrix to image file
def saveFloatImage(filename, image):
    image = image*255
    save_image = image.astype(np.uint8)
    cv2.imwrite(filename, save_image)

# current image subtracte resize image from Gaussion to get Laplacian image
def resizeSubImage(image, subimage, to_row=None, to_col=None):
    if (to_row==None) or (to_col==None):
        to_row = image.shape[0]
        to_col = image.shape[1]
    return resize(image, to_row, to_col) - resize(subimage, to_row, to_col)


def pyramid5(filepath):
    filename = filepath.split('/')[-1].split('.')[0]
    img = cv2.imread(filepath,0)
    img = img/255
    row,col = img.shape   
    
    img_l1 = pyramidDown(img)
    img_l1_gr = resize(img_l1, row, col)
    img_l1_lr = resizeSubImage(img, img_l1, row, col)
    cv2.imshow("Image Pyramid Down",img_l1_gr)
    saveFloatImage(filename+"_g1.png", img_l1_gr)
    cv2.imshow("Laplacian", img_l1_lr)
    saveFloatImage(filename+"_l1.png", img_l1_lr)
    cv2.imshow("Image Spectrum H", logSpectrum(img_l1_lr))
    cv2.imshow("Image Spectrum L", logSpectrum(img_l1_gr))
    saveFloatImage(filename+"_sl1.png", logSpectrum(img_l1_lr))
    saveFloatImage(filename+"_sg1.png", logSpectrum(img_l1_gr))
    cv2.waitKey()

    img_l2 = pyramidDown(img_l1)
    img_l2_gr = resize(img_l2, row, col)
    img_l2_lr = resizeSubImage(img_l1, img_l2, row, col)
    cv2.imshow("Image Pyramid Down", img_l2_gr)
    saveFloatImage(filename+"_g2.png", img_l2_gr)
    cv2.imshow("Laplacian",img_l2_lr)
    saveFloatImage(filename+"_l2.png", img_l2_lr)
    cv2.imshow("Image Spectrum H", logSpectrum(img_l2_lr))
    cv2.imshow("Image Spectrum L", logSpectrum(img_l2_gr))
    saveFloatImage(filename+"_sl2.png", logSpectrum(img_l2_lr))
    saveFloatImage(filename+"_sg2.png", logSpectrum(img_l2_gr))
    cv2.waitKey()
    
    img_l3 = pyramidDown(img_l2)
    img_l3_gr = resize(img_l3, row, col)
    img_l3_lr = resizeSubImage(img_l2, img_l3, row, col)
    cv2.imshow("Image Pyramid Down", img_l3_gr)
    saveFloatImage(filename+"_g3.png", img_l3_gr)
    cv2.imshow("Laplacian", img_l3_lr)
    saveFloatImage(filename+"_l3.png", img_l3_lr)
    cv2.imshow("Image Spectrum H", logSpectrum(img_l3_lr))
    cv2.imshow("Image Spectrum L", logSpectrum(img_l3_gr))
    saveFloatImage(filename+"_sl3.png", logSpectrum(img_l3_lr))
    saveFloatImage(filename+"_sg3.png", logSpectrum(img_l3_gr))
    cv2.waitKey()
    
    img_l4 = pyramidDown(img_l3)
    img_l4_gr = resize(img_l4, row, col)
    img_l4_lr = resizeSubImage(img_l3, img_l4, row, col)+0.5
    cv2.imshow("Image Pyramid Down", img_l4_gr)
    saveFloatImage(filename+"_g4.png", img_l4_gr)
    cv2.imshow("Laplacian", img_l4_lr)
    saveFloatImage(filename+"_l4.png", img_l4_lr)
    cv2.imshow("Image Spectrum H", logSpectrum(img_l4_lr))
    cv2.imshow("Image Spectrum L", logSpectrum(img_l4_gr))
    saveFloatImage(filename+"_sl4.png", logSpectrum(img_l4_lr))
    saveFloatImage(filename+"_sg4.png", logSpectrum(img_l4_gr))
    cv2.waitKey()

    img_l5 = pyramidDown(img_l4)
    img_l5_gr = resize(img_l5, row, col)
    img_l5_lr = resizeSubImage(img_l4, img_l5, row, col)+0.5
    cv2.imshow("Image Pyramid Down", img_l5_gr)
    saveFloatImage(filename+"_g5.png", img_l5_gr)
    cv2.imshow("Laplacian", img_l5_lr)
    saveFloatImage(filename+"_l5.png", img_l5_lr)
    cv2.imshow("Image Spectrum H", logSpectrum(img_l5_lr))
    cv2.imshow("Image Spectrum L", logSpectrum(img_l5_gr))
    saveFloatImage(filename+"_sl5.png", logSpectrum(img_l5_lr))
    saveFloatImage(filename+"_sg5.png", logSpectrum(img_l5_gr))
    cv2.waitKey()
    cv2.destroyAllWindows()


def pyramid5RGB(filepath):
    filename = filepath.split('/')[-1].split('.')[0]
    img = cv2.imread(filepath)
    img = img/255
    row,col,channel = img.shape
    
    img_l1 = pyramidDownRGB(img)
    img_l1_gr = img_l1
    img_l1_lr = resizeSubImage(img, img_l1)+0.5
    cv2.imshow("Image Pyramid Down", img_l1_gr)
    cv2.imshow("Laplacian", img_l1_lr)
    saveFloatImage(filename+"_g1c.png", img_l1_gr)
    saveFloatImage(filename+"_l1c.png", img_l1_lr)
    cv2.waitKey()
    cv2.destroyAllWindows()

    img_l2 = pyramidDownRGB(img_l1)
    img_l2_gr = img_l2
    img_l2_lr = resizeSubImage(img_l1, img_l2)+0.5
    cv2.imshow("Image Pyramid Down", img_l2_gr)
    cv2.imshow("Laplacian", img_l2_lr)
    saveFloatImage(filename+"_g2c.png", img_l2_gr)
    saveFloatImage(filename+"_l2c.png", img_l2_lr)
    cv2.waitKey()
    cv2.destroyAllWindows()

    img_l3 = pyramidDownRGB(img_l2)
    img_l3_gr = img_l3
    img_l3_lr = resizeSubImage(img_l2, img_l3)+0.5
    cv2.imshow("Image Pyramid Down", img_l3_gr)
    cv2.imshow("Laplacian", img_l3_lr)
    saveFloatImage(filename+"_g3c.png", img_l3_gr)
    saveFloatImage(filename+"_l3c.png", img_l3_lr)
    cv2.waitKey()
    cv2.destroyAllWindows()

    img_l4 = pyramidDownRGB(img_l3)
    img_l4_gr = img_l4
    img_l4_lr = resizeSubImage(img_l3, img_l4)+0.5
    cv2.imshow("Image Pyramid Down", img_l4_gr)
    cv2.imshow("Laplacian", img_l4_lr)
    saveFloatImage(filename+"_g4c.png", img_l4_gr)
    saveFloatImage(filename+"_l4c.png", img_l4_lr)
    cv2.waitKey()
    cv2.destroyAllWindows()

    img_l5 = pyramidDownRGB(img_l4)
    img_l5_gr = img_l5
    img_l5_lr = resizeSubImage(img_l4, img_l5)+0.5
    cv2.imshow("Image Pyramid Down", img_l5_gr)
    cv2.imshow("Laplacian", img_l5_lr)
    saveFloatImage(filename+"_g5c.png", img_l5_gr)
    saveFloatImage(filename+"_l5c.png", img_l5_lr)
    cv2.waitKey()
    cv2.destroyAllWindows()

    
def main(filepath):
    pyramid5(filepath)
    pyramid5RGB(filepath)

if __name__ == "__main__":
    filepath = './/data//cat.bmp'
    main(filepath)
