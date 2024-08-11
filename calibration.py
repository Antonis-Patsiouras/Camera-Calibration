import subprocess
import cv2
import numpy as np
import glob

# Καθορισμός του μεγέθους του πίνακα σκακιέρας
chessboard_size = (9, 6)
# Καθορισμός των κριτηρίων τερματισμού για το υποπίνακα
criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

# Προετοιμασία των αντικειμένων (0,0,0), (1,0,0), (2,0,0) ... , (8,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Λίστες για την αποθήκευση των σημείων αντικειμένου και των σημείων εικόνας από όλες τις εικόνες
objpoints = []  # 3d σημεία πραγματικού κόσμου
imgpoints = []  # 2d σημεία στην εικόνα

# Φόρτωση των εικόνων
images = glob.glob(r"C:\Users\apats\OneDrive\Documents\MSc Robotics\P201_1\Project2\images\*")
fname = (r"C:\Users\apats\OneDrive\Documents\MSc Robotics\P201_1\Project2\images\*")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Εύρεση των γωνιών της σκακιέρας
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # Αν βρεθούν οι γωνίες, προσθήκη σημείων αντικειμένου και σημείων εικόνας
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Εμφάνιση των γωνιών
        img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        np.save(r"C:\Users\apats\OneDrive\Documents\MSc Robotics\P201_1\Project2\camera\img.jpg", img)
        cv2.imshow('img', img)
        cv2.waitKey(500)
cv2.destroyAllWindows()

# Βαθμονόμηση της κάμερας
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Πίνακας εσωτερικών παραμέτρων (mtx):")
print(mtx)
print("\nΠαράμετροι παραμόρφωσης (dist):")
print(dist)
np.save(r"C:\Users\apats\OneDrive\Documents\MSc Robotics\P201_1\Project2\camera\ret", ret)
np.save(r"C:\Users\apats\OneDrive\Documents\MSc Robotics\P201_1\Project2\camera\mtx", mtx)
np.save(r"C:\Users\apats\OneDrive\Documents\MSc Robotics\P201_1\Project2\camera\dist", dist)
np.save(r"C:\Users\apats\OneDrive\Documents\MSc Robotics\P201_1\Project2\camera\rvecs", rvecs)
np.save(r"C:\Users\apats\OneDrive\Documents\MSc Robotics\P201_1\Project2\camera\tvecs", tvecs)

# Διόρθωση παραμορφωμένης εικόνας (παράδειγμα)
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Διόρθωση εικόνας
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Περικοπή εικόνας
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imshow('Undistorted Image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

subprocess.run(["python", "code.py"])