import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Loading of 2 images
img1 = cv2.imread(r"your_image_path", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r"your_image_path", cv2.IMREAD_GRAYSCALE)

# Εξαγωγή χαρακτηριστικών σημείων με τη μέθοδο SIFT
sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# BFMatcher για να ταιριάξετε τα χαρακτηριστικά σημεία
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Ταξινομηση των ταιριασματων με βάση την απόσταση
matches = sorted(matches, key=lambda x: x.distance)

# Εξαγωγή των σημείων ταιριάσματος
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

# Υπολογισμός του Βασικού Πίνακα με χρήση RANSAC
fundamental_matrix, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# Υπολογισμός του Ουσιώδους Πίνακα με χρήση RANSAC
# Φόρτωση των εσωτερικών παράμετρων της κάμερας 
camera_matrix = np.load(r"your_path")
dist_coeffs = np.load(r"your_path")

essential_matrix, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

print("Fundamental Matrix:\n", fundamental_matrix)
print("Essential Matrix:\n", essential_matrix)

# Αποσύνθεση του Ουσιώδους Πίνακα για τον υπολογισμό του πίνακα στροφής και του διανύσματος μετατόπισης
_, R, T, mask = cv2.recoverPose(essential_matrix, pts1, pts2, cameraMatrix=camera_matrix)

print("Rotation Matrix:\n", R)
print("Translation Vector:\n", T)

# Σχεδίαση των επιπολικών γραμμών στις εικόνες
def draw_epipolar_lines(img1, img2, lines, pts1, pts2):
    ''' Draw the epipolar lines on the images '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    for r, pts1, pts2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        #img1 = cv2.circle(img1, tuple(pts1), 5, color, -1)
        #img2 = cv2.circle(img2, tuple(pts2), 5, color, -1)
    return img1, img2

# Υπολογισμός των επιπολικών γραμμών
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, fundamental_matrix)
lines1 = lines1.reshape(-1, 3)
img1_epilines, img2_points = draw_epipolar_lines(img1, img2, lines1, pts1, pts2)

lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, fundamental_matrix)
lines2 = lines2.reshape(-1, 3)
img2_epilines, img1_points = draw_epipolar_lines(img2, img1, lines2, pts2, pts1)

# Εμφάνιση των αποτελεσμάτων
cv2.imshow('Epilines in Image 1', img1_epilines)
cv2.imshow('Epilines in Image 2', img2_epilines)


# Υπολογισμός των προβολικών πινάκων
P1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))  # Προβολικός πίνακας για την πρώτη κάμερα
P1 = camera_matrix @ P1

P2 = np.hstack((R, T))  # Προβολικός πίνακας για τη δεύτερη κάμερα
P2 = camera_matrix @ P2

print("Projection Matrix for Camera 1:\n", P1)
print("Projection Matrix for Camera 2:\n", P2)

# Τριγωνισμός των σημείων για να βρεθούν οι 3D συντεταγμένες
points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

# Μετατροπή των ομογενών συντεταγμένων σε Καρτεσιανές
points_3d = points_4d_hom[:3] / points_4d_hom[3]

# Προβολή των 3D σημείων σε ένα γράφημα
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points_3d[0], points_3d[1], points_3d[2], c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

# Ανορθώση των εικόνων
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(camera_matrix, dist_coeffs, camera_matrix, dist_coeffs, img1.shape[::-1], R, T, alpha=0)

map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R1, P1, img1.shape[::-1], cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R2, P2, img2.shape[::-1], cv2.CV_32FC1)

rectified_img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
rectified_img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

# Υπολογισμός του πυκνού χάρτη βάθους με χρήση του αλγορίθμου SGBM
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16,
    blockSize=15,
    P1=8 * 3 * 15 ** 2,
    P2=32 * 3 * 15 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

disparity_map = stereo.compute(rectified_img1, rectified_img2)

# Προβολή του χάρτη παραλλάξεων σε εικόνα αποχρώσεων του γκρι
plt.imshow(disparity_map, 'gray')
plt.colorbar()
plt.show()
