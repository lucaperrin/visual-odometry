import cv2
import numpy as np
import glob

def calibrate_camera(chessboard_size, square_size, image_path_pattern, output_file="camera_calib_params.npz"):

    # Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (chessboard_size[0]-1, chessboard_size[1]-1,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Load calibration images
    images = glob.glob(image_path_pattern)

    accepted_images = 0
    rejected_images = 0

    # Loop all images in folder
    for image_file in images:
        img = cv2.imread(image_file)  # Read the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If found, add object points, image points (after refining them)
        if ret:
            accepted_images += 1 
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(100)  # Display the image for 100 ms
        else:
            rejected_images += 1

    cv2.destroyAllWindows()  # Close all OpenCV windows

    # Calibrate the camera using OpenCV builtin function
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Print the intrinsic matrix and distortion coefficients
    print(" \nIntrinsic Matrix (K):")
    print(camera_matrix)

    # print("\n Distortion Coefficients (P):")
    # print(dist_coeffs)

    print(f"\n Accepted images: {accepted_images}")
    print(f"Rejected images: {rejected_images}")

    # Save the intrinsic matrix and distortion coefficients
    np.savez(output_file, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"File {output_file} saved successfully.")

    return camera_matrix, accepted_images, rejected_images