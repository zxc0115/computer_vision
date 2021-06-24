import os
os.chdir('C:\\Users\\zxc26\\Desktop\\CV2019_HW1')
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from numpy.linalg import inv , svd ,cholesky
from PIL import Image
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
pts_num = 10
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data/*.jpg')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
idx=4
fname='data\\0004.jpg'
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)
    corners = np.squeeze(corners)
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        #plt.imshow(img)


#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
'''
img_size = (img.shape[1], img.shape[0])
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
np.dot(mtx,np.hstack((cv2.Rodrigues(rvecs[0])[0],Tr[0])))
np.dot(cv2.Rodrigues(rvecs[0])[0].T , cv2.Rodrigues(rvecs[0])[0])
Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
_rotation = np.stack([cv2.Rodrigues(v)[0] for v in Vr])
_extrinsic = np.concatenate([_rotation[:,:,0:2], Tr], axis=2)
_homographies = np.stack([mtx.dot(_ext) for _ext in _extrinsic])
for i in range (0,10):
    _homographies[i] /= _homographies[i][-1][-1]
np.dot(homography[0][]np.dot(inv(mtx).T, inv(mtx)))
for i in range(0,pts_num):
    print(np.dot(np.dot(homography[i][:,0],real_B),homography[i][:,1]))
for i in range(0,pts_num):
    print(np.dot(np.dot(homography[i][:,0],np.dot(inv(mtx).T, inv(mtx))),homography[i][:,1]))
'''
# Write your code here

def get_v(k, l, h):
	v_kl = np.array([h[k, 0]*h[l, 0],
			h[k, 0]*h[l, 1] + h[k, 1]*h[l, 0],
			h[k, 2]*h[l, 0] + h[k, 0]*h[l, 2],
            h[k, 1]*h[l, 1],
			h[k, 2]*h[l, 1] + h[k, 1]*h[l, 2],
			h[k, 2]*h[l, 2]], dtype=float)
	assert v_kl.shape == (6,), v_kl.shape
	return v_kl
'''
def find_B(all_H):
    K_rank = all_H[0].shape[0]
    pts_num = len(all_H)
    parameter_num = int(K_rank*(K_rank + 1) / 2)
    print(parameter_num)
    V = np.zeros([2 * pts_num, parameter_num])
    for i in range (0,pts_num):
        
        #h1_h2_T = all_H[i][:,0].reshape(K_rank,1) * all_H[i][:,1]
        #h1_minus_h2 = all_H[i][:,0].reshape(K_rank,1) * all_H[i][:,0] - all_H[i][:,1].reshape(K_rank,1) * all_H[i][:,1]
        #tmp = h1_h2_T + h1_h2_T.T
        #tmp1 = h1_minus_h2 + h1_minus_h2.T
        #rank = 2*i
        V_dim = 0
        for j in range (0,K_rank):
            for k in range (j,K_rank):
                if j == k:
                    V[rank][V_dim] = tmp[j][k] / 2
                    V[rank + 1][V_dim] = tmp1[j][k] / 2
                else:
                    V[rank][V_dim] = tmp[j][k]
                    V[rank + 1][V_dim] = tmp1[j][k]
                V_dim += 1
    print(V)
    u , s ,vh = svd(V)
    #print(s)
    #print(vh)
    B = vh[2]
    #B = find_H(V[:,1:],-V[:,0])
    return B
#homography[0]
'''
def find_B(all_H):
    K_rank = all_H[0].shape[0]
    #pts_num = len(all_H)
    parameter_num = int(K_rank*(K_rank + 1) / 2)
    V = np.zeros([2 * pts_num, parameter_num])
    for i in range (0,pts_num):
        Vdim = 2*i
        V[Vdim] = get_v(1, 0, all_H[i].T)
        V[Vdim+1] = get_v(0, 0, all_H[i].T) - get_v(1, 1, all_H[i].T)
    u , s ,vh = svd(V)
    #print(V)
    print(vh)
    print('---------------------------------------')
    print(s)
    B = vh[-1]
    return B

def find_H(objp , imgpoint):
    P = np.zeros([2 * objp.shape[0],9])
    for i in range (0, objp.shape[0]):
        tmp_row = np.zeros([2,9])
        for j in range (0,2):
            tmp_row[j][3 * j : 3 * j + 3] = objp[i]
            tmp_row[j][6:9] = imgpoint[i][j] * -objp[i]
            print(tmp_row)
        P[2*i:2*i+2] = tmp_row
    #print(P)
    M = svd(P)[2][-1]
    H = M.reshape(3,3)
    return H
objp[:,2]=1
homography=[]
h = []
imgpoint = np.zeros([imgpoints[0].shape[0],imgpoints[0].shape[1]+1])
one = np.zeros([objp.shape[0],1])
imgpoint[:,2]=1
#one[:,0] = 1
np.hstack((objp , one))
for i in range(0,pts_num):
    imgpoint[:,:2]=imgpoints[i]
    H = find_H(objp, imgpoints[i])
    homography.append(H/H[-1][-1])
    #homography.append(cv2.findHomography(objp, imgpoints[i])[0])
B=-find_B(homography)
#B[3]=-B[3]
#B[4]=-B[4]
real_B = np.zeros(homography[0].shape)
real_B[0][0] = 1
dim = 0
for i in range(0,homography[0].shape[0]):
    for j in range(i,homography[0].shape[1]):
        real_B[i][j] = B[dim]
        real_B[j][i] = B[dim]
        dim += 1
K_inv = cholesky(real_B).T
K = inv(K_inv)
mtx, rvecs, tvecs=K,[],[]
for i in range (0,pts_num):
    lamb1=1/np.sqrt(np.sum(np.dot(K_inv,homography[i][:,0])**2))
    lamb2=1/np.sqrt(np.sum(np.dot(K_inv,homography[i][:,1])**2))
    r1 = np.dot(K_inv,homography[i][:,0]) * lamb1
    r2 = np.dot(K_inv,homography[i][:,1]) * lamb1
    r3=np.cross(r1,r2)
    print(np.dot(r1,r2))
    t = np.dot(K_inv,homography[i][:,2]) * lamb1 
    rvecs.append(cv2.Rodrigues(np.vstack([r1,r2,r3]).T)[0])
    tvecs.append(t.reshape(3,1))
Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
'''svd(homography[0])[2][8]
cv2.findHomography(objp, imgpoints[0])
homography[0]
(real_B[0][1]*real_B[0][2]-real_B[0][0]*real_B[1][2])/(real_B[0][0]*real_B[1][1]-real_B[0][1]**2)
#find_H(objp, imgpoint)

np.dot(np.dot(homography[0][:,0],homography[1]),homography[0][:,1].reshape(2,1))
np.sum(homography[0][:,0].reshape(2,1) * homography[0][:,1] * homography[1])

np.dot(np.dot(homography[0][1].T,real_B),homography[0][0])
np.array(imgpoints).shape
'''
# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
camera_matrix = mtx
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""
