import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random
import math

# =========================================================================================================
# Function Definition

def normalize_coordinates(pts):
    # Convert list to numpy array if necessary
    if type(pts) == list:
        pts = np.array(pts)

    # Compute the centroid of the points
    centroid = np.mean(pts, axis=0)

    # Compute the scale factor
    std_dev = np.std(pts, axis=0).mean()
    scale = np.sqrt(2) / std_dev

    # Construct the normalization matrix
    T = np.array([[scale, 0, -scale*centroid[0]],
                  [0, scale, -scale*centroid[1]],
                  [0, 0, 1]])

    # Normalize the coordinates
    pts_norm = np.matmul(T, pts.T).T

    return pts_norm, T

def EstimateEssentialMatrix(F,K_left,K_right):
    E=K_left.T @ F @K_right

    u, s, vh = np.linalg.svd(E, full_matrices=True)
    S_rank2=np.array([[1,0,0],
                      [0,1,0],
                      [0,0,0]])
    E_rank2=u@S_rank2@vh
    return E_rank2

def ExtractRotationandTranslation(pts1,pts2,K,E):
    u, s, vh = np.linalg.svd(E, full_matrices=True)
    W=np.array([[0,-1,0],
                [1,0,0],
                [0,0,1]])
    
    C1=u[:,-1]
    R1=u@W@vh
    C2=-u[:,-1]
    R2=u@W@vh
    C3=u[:,-1]
    R3=u@W.T@vh
    C4=-u[:,-1]
    R4=u@W.T@vh

    if np.linalg.det(R1) < 0:
        C1 = -C1
        R1 = -R1
    if np.linalg.det(R2) < 0:
        C2 = -C2
        R2 = -R2
    if np.linalg.det(R3) < 0:
        C3 = -C3
        R3 = -R3
    if np.linalg.det(R4) < 0:
        C4 = -C4
        R4 = -R4

    # C, R=ChooseCorrectConfiguration(pts1,pts2,K,C1,R1,C2,R2,C3,R3,C4,R4)
    Translation= -R1@C1
    return Translation,R1

def ChooseCorrectConfiguration(pts1,pts2,K,C1,R1,C2,R2,C3,R3,C4,R4):
    P1=K@R1@np.array([[1,0,0,-C1[0]],
                     [0,1,0,-C1[1]],
                     [0,0,1,-C1[2]]])
    
    P2=K@R2@np.array([[1,0,0,-C2[0]],
                     [0,1,0,-C2[1]],
                     [0,0,1,-C2[2]]])
    
    P3=K@R3@np.array([[1,0,0,-C3[0]],
                     [0,1,0,-C3[1]],
                     [0,0,1,-C3[2]]])
    
    P4=K@R4@np.array([[1,0,0,-C4[0]],
                     [0,1,0,-C4[1]],
                     [0,0,1,-C4[2]]])
   
    # Compute 3D points using triangulation for all four configurations
    X1 = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
    X2 = cv.triangulatePoints(P1, P3, pts1.T, pts2.T)
    X3 = cv.triangulatePoints(P1, P4, pts1.T, pts2.T)
    X4 = cv.triangulatePoints(P2, P3, pts1.T, pts2.T)

    # Check which configuration produces 3D points with positive depth
    depth1 = X1[:, 2] > 0
    depth2 = X2[:, 2] > 0
    depth3 = X3[:, 2] > 0
    depth4 = X4[:, 2] > 0

    # Determine which configuration is correct
    num_depth1 = np.sum(depth1)
    num_depth2 = np.sum(depth2)
    num_depth3 = np.sum(depth3)
    num_depth4 = np.sum(depth4)

    max_num_depth = max(num_depth1, num_depth2, num_depth3, num_depth4)

    if max_num_depth == num_depth1:
        C = C1
        R = R1
    elif max_num_depth == num_depth2:
        C = C2
        R = R2
    elif max_num_depth == num_depth3:
        C = C3
        R = R3
    else:
        C = C4
        R = R4

    # Correct the camera pose if necessary
    if np.linalg.det(R) < 0:
        C = -C
        R = -R

    return C, R

def ComputeFundamentalMatrix(pts1,pts2):
    A=np.empty((0,9), float)
    for i in range(len(pts1)):
        A = np.append(A, [np.array([pts1[i][0]*pts2[i][0],pts1[i][0]*pts2[i][1],pts1[i][0],pts1[i][1]*pts2[i][0],pts1[i][1]*pts2[i][1],pts1[i][1],pts2[i][0],pts2[i][1],1])], 0)

    # Compute F
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    v=np.transpose(vh)
    F_col=v[:,-1]

    F=np.array([[F_col[0],F_col[1],F_col[2]],
                [F_col[3],F_col[4],F_col[5]],
                [F_col[6],F_col[7],F_col[8]]])
    
    u_f, s_f, vh_f = np.linalg.svd(F, full_matrices=True)
    S_rank2=np.array([[s_f[0],0,0],
                [0,s_f[1],0],
                [0,0,0]])
    F_rank2=u_f@S_rank2@vh_f
    
    final_F=F_rank2/F_rank2[2][2]

    return final_F

def ComputeFundamentalMatrix_RANSAC(pts1_dnorm,pts2_dnorm):
    # pts1,T1=normalize_coordinates(pts1_dnorm)
    # pts2,T2=normalize_coordinates(pts2_dnorm)

    pts1=pts1_dnorm
    pts2=pts2_dnorm

    print(pts1.shape)
    N=1000000000000000000000000000000000000000
    p=0.99
    sample_count=0
    num_inliers=[]
    F_values=[]
    threshold=0.05

    while N> sample_count:
        # Choose 8 point matches randomly
        sample_indices=[]
        while len(sample_indices)<8:
            id = random.randrange(pts1.shape[0]-1)
            if not(id in sample_indices):
                sample_indices.append(id)
    
        
        # Find fundamental using 8 matches
        pts1_subset= []
        pts2_subset= []

        for i in range(len(sample_indices)):
            pts1_subset.append(pts1[sample_indices[i]])
            pts2_subset.append(pts2[sample_indices[i]])
    
        F=ComputeFundamentalMatrix(pts1_subset,pts2_subset)

        # print(pts1_subset.shape)

        inlier_count=0
       
        for i in range(pts1.shape[0]):
            x1=np.array([[pts1[i][0]],
                         [pts1[i][1]],
                         [1]])
            x2=np.array([[pts2[i][0]],
                         [pts2[i][1]],
                         [1]])
            val=(x1.T@F)@x2
        
            if abs(val) < threshold:
                inlier_count+=1
                # pts1_inliers.append(x1)
                # pts2_inliers.append(x2)
    
        num_inliers.append(inlier_count)
        F_values.append(F)
        e=1-(inlier_count/pts1.shape[0])

        denominator=math.log(1-(1-e)**8)
        if not denominator==0: 
            N=math.log(1-p)/denominator

        sample_count+=1

    sol_index=num_inliers.index(max(num_inliers))
    print(num_inliers[sol_index])
    final_F=F_values[sol_index]

    pts1_inliers=[]
    pts2_inliers=[]
    for i in range(pts1.shape[0]):
            x1=np.array([[pts1[i][0]],
                         [pts1[i][1]],
                         [1]])
            x2=np.array([[pts2[i][0]],
                         [pts2[i][1]],
                         [1]])
            val=(x1.T@F)@x2
        
            if abs(val) < threshold:
                pts1_inliers.append(pts1[i])
                pts2_inliers.append(pts2[i])

    pts1_inliers = np.int32(pts1_inliers)
    pts2_inliers = np.int32(pts2_inliers)
    # print(pts1_inliers.shape)
    print(np.linalg.matrix_rank(final_F))
    return final_F,pts1_inliers,pts2_inliers

def drawlines(img1,img2,lines,pts1_inliers,pts2_inliers):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1_inliers,pts2_inliers):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# =========================================================================================================
# Main

img1 = cv.imread('chess/im0.png', cv.IMREAD_GRAYSCALE)  #queryimage # left image
img2 = cv.imread('chess/im1.png', cv.IMREAD_GRAYSCALE) #trainimage # right image
K=np.array([[1758.23, 0, 829.15],
            [0, 1758.23, 552.78],
            [0,0,1]])
baseline=97.99

# Feature detection
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
pts1 = []
pts2 = []

# ratio test 
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# Compute fundamental matrix
F1,pts1_inliers,pts2_inliers=ComputeFundamentalMatrix_RANSAC(pts1,pts2)
F1, mask = cv.findFundamentalMat(pts1,pts2,cv.RANSAC)
print("Fundamemtal Matrix")
print(F1)

pts1_inliers = pts1[mask.ravel()==1]
pts2_inliers = pts2[mask.ravel()==1]

myE=EstimateEssentialMatrix(F1,K,K)
print("Essential Matrix")
print(myE)

trans,R=ExtractRotationandTranslation(pts1,pts2,K,myE)
print("Translation")
print(trans)
print("Rotation")
print(R)

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2_inliers.reshape(-1,1,2), 2,F1)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1_inliers,pts2_inliers)


# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1_inliers.reshape(-1,1,2), 1,F1)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2_inliers,pts1_inliers)

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

h1, w1 = img1.shape
h2, w2 = img2.shape
_, H1, H2 = cv.stereoRectifyUncalibrated(
    np.float32(pts1_inliers), np.float32(pts2_inliers), F1, imgSize=(w1, h1)
)

print("Homography Matrix 1 (H1)")
print(H1)
print("Homography Matrix 2 (H2)")
print(H2)

img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))
cv.imwrite("rectified_1.png", img1_rectified)
cv.imwrite("rectified_2.png", img2_rectified)


# Draw the rectified images
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
axes[0].imshow(img1_rectified, cmap="gray")
axes[1].imshow(img2_rectified, cmap="gray")
axes[0].axhline(100)
axes[1].axhline(100)
axes[0].axhline(200)
axes[1].axhline(200)
axes[0].axhline(300)
axes[1].axhline(300)
axes[0].axhline(400)
axes[1].axhline(400)
axes[0].axhline(500)
axes[1].axhline(500)
axes[0].axhline(600)
axes[1].axhline(600)
axes[0].axhline(700)
axes[1].axhline(700)
axes[0].axhline(800)
axes[1].axhline(800)
axes[0].axhline(900)
axes[1].axhline(900)
axes[0].axhline(1000)
axes[1].axhline(1000)
plt.suptitle("Rectified images")
plt.savefig("rectified_images.png")
plt.show()

disparity_image=np.zeros(img1_rectified.shape)
ndsp=40

block_size=11

print(img1_rectified.shape)

img1_rows=img1_rectified.shape[0]
img1_clms=img1_rectified.shape[1]
img2_clms=img2_rectified.shape[1]

start_id=int(block_size/2)

for i in range(start_id,img1_rows-start_id):
    # print(i)
    for j in range(start_id,img1_clms-start_id):
        left_image_window=img1_rectified[i-start_id:i+start_id+1,j-start_id:j+start_id+1]
        SSDiff=[]
        for j_right in range(j,min(j+ndsp,img2_clms-start_id)):
            right_image_window=img2_rectified[i-start_id:i+start_id+1,j_right-start_id:j_right+start_id+1]
            diff=abs(left_image_window-right_image_window)
            matching_score_ssd = np.sum(np.square(diff))
            SSDiff.append(matching_score_ssd)
            
        minpos = SSDiff.index(min(SSDiff))
        disparity_image[i,j]=minpos   


disparity_image=disparity_image*255/(np.max(disparity_image)-np.min(disparity_image))
# disparity_normalized_gray = cv.normalize(disparity_image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
disparity_color_map = cv.applyColorMap(disparity_image.astype(np.uint8), cv.COLORMAP_JET)
# save the resulting image
cv.imwrite('disparity_gray.png', disparity_image)
cv.imwrite('disparity_colored.png', disparity_color_map)


depth_image=np.zeros(disparity_image.shape)
# depth_image=np.where(disparity_image==0,0,baseline*K[0][0]/disparity_image)
mask = disparity_image > 0
depth_image[mask] = baseline*K[0][0]/disparity_image[mask]

depth_image=depth_image*255/(np.max(depth_image)-np.min(depth_image))
cv.imwrite('depth_gray.png', depth_image)
depth_color_map = cv.applyColorMap(depth_image.astype(np.uint8), cv.COLORMAP_JET)
cv.imwrite('depth_colored.png', depth_color_map)