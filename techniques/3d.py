import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pykalman import KalmanFilter
from scipy.signal import medfilt


def load_images(directory):
    """Load and sort image files from a directory."""
    image_files = sorted([f for f in os.listdir(directory) if f.endswith(('.png', '.TIF'))])
    if not image_files:
        raise FileNotFoundError(f"No image files found in directory: {directory}")
    return image_files

def filter_matches_with_ransac(pts1, pts2):
    """Filter matches using RANSAC on the fundamental matrix."""
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 1.0)
    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]
    return inliers1, inliers2

def process_image_pair(img1, img2, orb, flann, K, dist_coeffs, ratio_test, frame_index):
    """
    Process a pair of images, compute the essential matrix,
    and extract rotation and translation between them. Visualize keypoints and their movements.
    """

    #Undistort images
    # img1 = cv2.undistort(img1, K, dist_coeffs)
    # img2 = cv2.undistort(img2, K, dist_coeffs)

    #ORB descripteur
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    #flan Matcher
    matches = flann.knnMatch(des1, des2, k=2)

    #lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_test * n.distance:
            good_matches.append(m)

    #Matched Keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # RANSAC filtering
    pts1, pts2 = filter_matches_with_ransac(pts1, pts2)
    
    ## POSE
    E , mask = cv2.findEssentialMat(pts1, pts2, K)
    #E2 = K.T @ F @ K
    #print('E calculated with matrix operations: \n', E2, '\n with findEssentialMat:', E)

    # Comserve only inliers
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Visualize keypoints and movements
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow(f"Keypoints and Movements - Frame {frame_index}", img_matches)
    cv2.waitKey(100)  # Wait for 200ms to visualize the frame
    cv2.destroyAllWindows()  # Close any remaining OpenCV windows

    # Built-in OpenCV function to recover pose
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    if R is None or t is None:
        print(f"Pose recovery failed for frames {frame_index}")

    return R, t, pts1, pts2


def compute_trajectory(image_files, directory, orb, flann, K, dist_coeffs, ratio_test):
    """Compute the camera trajectory from a sequence of images."""
    pose_global = np.eye(4)
    #positions = [pose_global[:3, 3]]
    positions = []


    for i in range(len(image_files) - 1):
        img1_path = os.path.join(directory, image_files[i])
        img2_path = os.path.join(directory, image_files[i + 1])

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            print(f"Error loading images {image_files[i]} or {image_files[i + 1]}")
            continue

        R, t, _, _ = process_image_pair(img1, img2, orb, flann, K, dist_coeffs, ratio_test, i + 1)
        if R is None or t is None:
            print(f"Using previous pose for frame {i + 1} due to failure")
            T = np.eye(4)  # Use identity transformation as a fallback
        else:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t.ravel()
            print('\nCurrent pose:\n', T)

        # # Update global pose
        pose_global = pose_global @ T
        positions.append((pose_global[0, 3], pose_global[1, 3], pose_global[2, 3]))


    return np.array(positions)

def remove_outliers(positions):
    """Remove outliers using a simple distance-based threshold."""
    median = np.median(positions, axis=0)
    distances = np.linalg.norm(positions - median, axis=1)
    threshold = np.percentile(distances, 90)  # Adjust as needed
    inliers = distances < threshold
    return positions[inliers]

def kalman_smooth(positions):
    """Smooth trajectory using a Kalman filter."""
    if positions.shape[1] != 3:
        raise ValueError("Positions must have 3 dimensions (x, y, z).")

    kf = KalmanFilter(initial_state_mean=positions[0], n_dim_obs=3)
    kf = kf.em(positions, n_iter=10)  # Estimate model parameters
    smoothed_positions, _ = kf.smooth(positions)
    return smoothed_positions


def smooth_trajectory(positions, num_points=500):
    """Smooth the trajectory with outlier removal and Kalman filtering."""
    # Step 1: Remove outliers
    positions = remove_outliers(positions)

    # Step 2: Interpolate for regular sampling
    t = np.linspace(0, 1, len(positions))
    t_smooth = np.linspace(0, 1, num_points)
    x_interp = interp1d(t, positions[:, 0], kind='cubic')
    y_interp = interp1d(t, positions[:, 1], kind='cubic')
    z_interp = interp1d(t, positions[:, 2], kind='cubic')

    interpolated_positions = np.vstack([x_interp(t_smooth), y_interp(t_smooth), z_interp(t_smooth)]).T

    # Step 3: Apply Kalman filter for smoothing
    smoothed_positions = kalman_smooth(interpolated_positions)

    return smoothed_positions

def generate_ground_truth(image_directory):
    if image_directory == 'images/circular/images':
        """Generate ground truth coordinates for a circle."""
        radius = 4  # diametre de 80cm
        step_degrees = 15  # Degrees
        angles = np.deg2rad(np.arange(0, 360, step_degrees))
        x = radius * np.cos(angles)
        y = radius * np.sin(angles) - radius
        z = np.zeros_like(x)  # Ajouter une composante z
        return np.vstack((x, y, z)).T
    
    elif image_directory == 'images/rectangular/images':
        # Dimensions du rectangle
        width = 1.34  # en mètres
        height = 1.5  # en mètres
        y = [0, -width, -width, 0, 0]
        x = [0, 0, -height, -height, 0]
        z = [0] * len(x)  # Ajouter une composante z
        return np.vstack((x, y, z)).T
    
    elif image_directory == 'images/linear/images':
        start_point = (0, 0)  # Starting point at the origin
        length = 3.5  # Convert cm to meters
        num_points = 14
        x = np.linspace(start_point[0], start_point[0] - length, num_points)
        y = np.full(num_points, start_point[1])  # Keep y constant for a horizontal line
        z = np.zeros(num_points)  # Ajouter une composante z
        return np.column_stack((x, y, z)).T
    

def plot_trajectories_with_ground_truth_3d(positions, smoothed_positions, ground_truth):
    """Plot the original, smoothed, and ground truth trajectories in 3D."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', label='Original Trajectory', alpha=0.6)

    # Ajouter des annotations pour les points de la trajectoire originale
    for i, (x, y, z) in enumerate(positions):
        ax.text(x, y, z, f'image{i + 1}', fontsize=8)

    # Plot smoothed trajectory
    ax.plot(smoothed_positions[:, 0], smoothed_positions[:, 1], smoothed_positions[:, 2], color='r', label='Smoothed Trajectory', alpha=0.8)

    # Plot ground truth
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], color='g', linestyle='--', label='Ground Truth Trajectory')

    # Labels and legend
    ax.set_title("3D Trajectory Comparison")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.grid()
    plt.show()


def main(image_directory,number_of_features,ratio_test):

    cv2.destroyAllWindows()  # Close any remaining OpenCV windows

    # Load camera intrinsics
    data = np.load('camera_calib_params.npz')
    K = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

    # Initialize ORB
    orb = cv2.ORB_create(nfeatures=number_of_features)

    # Initialize FLANN
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)  # FLANN_INDEX_LSH
    search_params = dict(checks=50)  # Number of trees to check
    flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    # Load images
    try:
        image_files = load_images(image_directory)
    except FileNotFoundError as e:
        print(e)
        return

    # Compute trajectory
    positions = compute_trajectory(image_files, image_directory, orb, flann, K, dist_coeffs, ratio_test)

    # Smooth the trajectory
    smoothed_positions = smooth_trajectory(positions)

    # Generate ground truth
    radius = 4  # diametre de 80cm
    step_degrees = 15  # Degrees
    ground_truth = generate_ground_truth(image_directory)

    # Plot both trajectories
    plot_trajectories_with_ground_truth_3d(positions, smoothed_positions, ground_truth)
