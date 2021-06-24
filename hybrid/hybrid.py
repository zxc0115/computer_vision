import os 
import cv2
import matplotlib.pyplot as plt
from ipdb import set_trace
import numpy as np
from math import exp
from scipy.spatial import distance
from scipy import signal
import scipy.misc
import time
np.set_printoptions(precision=3, suppress=True, threshold=100)

DATA_DIR = "./data/"

image_list = ["cat", "dog",
				"bicycle","motorcycle",
				"einstein", "marilyn", 
				"bird", "plane",
				"submarine", "fish"]
def dst_p2(a,b):
	return (a[0]-b[0])**2 + (a[1]-b[1])**2

def filter(img_f, D0, type, mode):
	"""Explanation
	Variable:
		img_f: 2D image in frequency domain
		D0: Cut-off frequency
		type: Low-pass or High-pass filter
		mode: Gaussian filter or Ideal filter
	Return value:
		img_f_fil: image that has been filtered
	"""
	assert type in ["LOW_PASS", "HIGH_PASS"] and mode in ["Gaussian", "Ideal"]
	shape = img_f.shape
	img_f_fil = np.empty(shape=shape, dtype=complex)
	center_x, center_y = shape[1]/2, shape[0]/2 #input img is in H,W format
	for x in range(shape[1]):
		for y in range(shape[0]):
			dst_sq = dst_p2((x,y), (center_x,center_y))#compute the distance square
			f = -1
			if mode == "Gaussian":
				if type == "LOW_PASS":
					f = exp(-dst_sq / (2 * (D0**2)))
				elif type == "HIGH_PASS":
					f = 1 - exp(-dst_sq / (2 * (D0**2)))
			elif mode == "Ideal":
				if type == "LOW_PASS":
					if dst_sq <= D0**2:
						f = 1
					else:
						f = 0
				elif type == "HIGH_PASS":
					if dst_sq > D0**2:
						f = 1
					else:
						f = 0
			img_f_fil[y,x] = img_f[y,x] * f
	return img_f_fil

def show_freq(img_f, title):
	fig, ax = plt.subplots()
	mag = abs(img_f)
	ax.imshow(np.log(mag), interpolation="none")
	ax.set_title('log(freq) of {0}'.format(title))
	plt.show()
def centralize(img):
	img[::2, ::2] = -img[::2, ::2] #Multiply every even row-colume pair by -1
	img[1::2, 1::2] = -img[1::2, 1::2] #Multiply every odd row-colume pair by -1
	return img
def img_hybrid():
	os.system("mkdir RGB")
	for i in range(0, 10, 2):
		print("Processing {0} pair .....".format(image_list[i]))
		
		##create directory
		os.chdir("./RGB")
		os.system("mkdir {0}".format(image_list[i]))
		os.chdir("..")
		
		img_1_org = cv2.imread(DATA_DIR + image_list[i] + ".bmp", cv2.IMREAD_COLOR) 
		img_2_org = cv2.imread(DATA_DIR + image_list[i+1] + ".bmp", cv2.IMREAD_COLOR)
		assert img_1_org.shape == img_2_org.shape, "imgs shapes are not equal"
		img_shape = img_1_org.shape
		
		img_1_f_org = np.empty(shape=img_shape, dtype=complex)
		img_2_f_org = np.empty(shape=img_shape, dtype=complex)
		for c in range(img_shape[2]): #Transform each channel of image to frequency domain
			img_1_ = img_1_org[:,:,c] #pick a certain color channel
			img_2_ = img_2_org[:,:,c] #pick a certain color channel
			img_1_ = centralize(img_1_) #centralize
			img_2_ = centralize(img_2_) #centralize
			img_1_f_org[:,:,c] = np.fft.fft2(img_1_)
			img_2_f_org[:,:,c] = np.fft.fft2(img_2_)
			
		for filter_mode in ["Ideal","Gaussian"]: #Choose which filter to use
			for D0 in [7,10,12,15,20,30]: #Choose which D0 to apply
				## RGB
				hybrid_img = np.empty(shape=img_shape)
				img_1_after = np.empty(shape=img_shape)
				img_2_after = np.empty(shape=img_shape)
				for c in range(img_shape[2]): #Process each channel
					##Get transformed values of each channel
					img_1_f_ = img_1_f_org[:,:,c]
					img_2_f_ = img_2_f_org[:,:,c]
					##Apply the filter in freqency domain
					img_1_f_fil_ = filter(img_f=img_1_f_, D0=D0, type="LOW_PASS", mode=filter_mode)
					img_2_f_fil_ = filter(img_f=img_2_f_, D0=D0, type="HIGH_PASS", mode=filter_mode)
					
					##Add up two images in freqency domain
					hybrid_img_f_ = img_1_f_fil_ + img_2_f_fil_
					##Transform back to space domain, pick the real part and add to certain channel of imgs
					img_1_after[:,:,c] = centralize(np.real(np.fft.ifft2(img_1_f_fil_)))
					img_2_after[:,:,c] = centralize(np.real(np.fft.ifft2(img_2_f_fil_)))
					hybrid_img[:,:,c] = centralize(np.real(np.fft.ifft2(hybrid_img_f_)))
				
				##save hybrid img (RGB)
				cv2.imwrite('./{3}/{2}/{1}_{0}.jpg'.format(D0, filter_mode, image_list[i],"RGB"), hybrid_img)
				cv2.imwrite('./{4}/{2}/{1}_{0}_{3}.jpg'.format(D0, filter_mode, image_list[i], image_list[i],"RGB"), img_1_after)
				cv2.imwrite('./{4}/{2}/{1}_{0}_{3}.jpg'.format(D0, filter_mode, image_list[i], image_list[i+1],"RGB"), img_2_after)
if __name__ == '__main__':
    img_hybrid()