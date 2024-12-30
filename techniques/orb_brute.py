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

def process_image_pair(img1, img2, orb, bf, K, dist_coeffs, ratio_test, frame_index):
    """
    Process a pair of images to estimate relative pose using feature matching.
    
    Steps:
    1. Detect and compute ORB features for both images.
    2. Match features using BRUTE with Lowe's ratio test.
    3. Filter matches with RANSAC to remove outliers.
    4. Compute essential matrix, rotation, and translation.
    5. Visualize the matched keypoints and movements.

    Parameters:
    - img1, img2: Pair of images to process.
    - orb: ORB detector instance.
    - flann: FLANN-based matcher.
    - K: Camera intrinsic matrix.
    - dist_coeffs: Distortion coefficients for undistortion (currently unused).
    - ratio_test: Lowe's ratio threshold for feature matching.
    - frame_index: Index of the current frame for visualization.

    Returns:
    - R: Rotation matrix between the two frames.
    - t: Translation vector between the two frames.
    - pts1, pts2: Inlier keypoints in the two frames.
    """
    # ORB descriptor
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Brute-Force Matcher
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_test * n.distance:
            good_matches.append(m)

    # Matched Keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # RANSAC filtering
    pts1, pts2 = filter_matches_with_ransac(pts1, pts2)

    # Pose estimation
    E, mask = cv2.findEssentialMat(pts1, pts2, K)

    # Consider only inliers
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Visualize keypoints and movements
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow(f"Keypoints and Movements - Frame {frame_index}", img_matches)
    cv2.waitKey(100)  # Wait for 100ms to visualize the frame
    cv2.destroyAllWindows()  # Close any remaining OpenCV windows

    # Recover pose
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    if R is None or t is None:
        print(f"Pose recovery failed for frames {frame_index}")

    return R, t, pts1, pts2

def compute_trajectory(image_files, directory, orb, bf, K, dist_coeffs, ratio_test):
    """
    Compute the camera trajectory from a sequence of images.

    Parameters:
    - image_files: List of image file paths.
    - directory: Directory containing the image files.
    - orb: ORB detector instance.
    - flann: FLANN-based matcher.
    - K: Camera intrinsic matrix.
    - dist_coeffs: Distortion coefficients for undistortion.
    - ratio_test: Lowe's ratio threshold for feature matching.

    Returns:
    - positions: List of 2D coordinates representing the estimated trajectory.
    - orientations: List of rotation matrices for each frame.
    """
    pose_global = np.eye(4)
    positions = []
    orientations = []

    for i in range(len(image_files) - 1):
        img1_path = os.path.join(directory, image_files[i])
        img2_path = os.path.join(directory, image_files[i + 1])

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            print(f"Error loading images {image_files[i]} or {image_files[i + 1]}")
            continue

        R, t, _, _ = process_image_pair(img1, img2, orb, bf, K, dist_coeffs, ratio_test, i + 1)
        if R is None or t is None:
            print(f"Using previous pose for frame {i + 1} due to failure")
            T = np.eye(4)  # Use identity transformation as a fallback
        else:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t.ravel()

        # Update global pose
        pose_global = pose_global @ T
        positions.append((pose_global[0, 3], pose_global[1, 3]))
        orientations.append(pose_global[:3, :3])

    return np.array(positions), orientations

def remove_outliers(positions):
    """Remove outliers using a simple distance-based threshold."""
    median = np.median(positions, axis=0)
    distances = np.linalg.norm(positions - median, axis=1)
    threshold = np.percentile(distances, 90)  # Adjust as needed
    inliers = distances < threshold
    return positions[inliers]

def kalman_smooth(positions):
    """Smooth trajectory using a Kalman filter."""
    kf = KalmanFilter(initial_state_mean=positions[0], n_dim_obs=2)
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
    interpolated_positions = np.vstack([x_interp(t_smooth), y_interp(t_smooth)]).T

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
        return np.vstack((x, y)).T
    
    elif image_directory == 'images/rectangular/images':
        # Dimensions du rectangle
        width = 3  # en mètres
        height = 3  # en mètres
        y = [0, -width, -width, 0, 0]
        x = [height, height, -height, -height, height]
        return np.vstack((x, y)).T
    
    
    elif image_directory == 'images/linear/images':
        start_point = (0, 0)  # Starting point at the origin
        length = 3.5  # Convert cm to meters
        num_points = 14
        x = np.linspace(start_point[0], start_point[0] - length, num_points)
        y = np.full(num_points, start_point[1])  # Keep y constant for a horizontal line
        return np.column_stack((x, y))
    else:
        x = 0
        y = 0
        return np.column_stack((x, y))

def plot_trajectories_with_orientations(positions, orientations, smoothed_positions, ground_truth):
    """Plot the trajectory with orientations (x and y axes), smoothed trajectory, and ground truth."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot original trajectory
    ax.plot(positions[:, 0], positions[:, 1], marker='o', label='Estimated Trajectory', alpha=0.6)

    # Ajouter des flèches pour les axes x et y locaux
    for i, (pos, orientation) in enumerate(zip(positions, orientations)):
        # Axe x
        direction_x = orientation[:2, 0]
        ax.arrow(pos[0], pos[1], direction_x[0] * 0.1, direction_x[1] * 0.1, 
                 head_width=0.05, head_length=0.08, fc='blue', ec='blue', alpha=0.7, label='Camera X-axis' if i == 0 else "")

        # Axe y
        direction_y = orientation[:2, 1]
        ax.arrow(pos[0], pos[1], direction_y[0] * 0.1, direction_y[1] * 0.1, 
                 head_width=0.05, head_length=0.08, fc='red', ec='red', alpha=0.7, label='Camera Y-axis' if i == 0 else "")

        ax.annotate(f"Img {i + 1}", (pos[0], pos[1]), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8)


    # Plot smoothed trajectory
    ax.plot(smoothed_positions[:, 0], smoothed_positions[:, 1], color='r', label='Smoothed Trajectory', alpha=0.8)

    # Plot ground truth
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], color='g', linestyle='--', label='Ground Truth Trajectory')

    # Labels and legend
    ax.set_title("Orb-Brute : Estimated, Smoothed and Ground-Truth Trajectories")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.axis('equal')  # Ensure the aspect ratio is equal for proper comparison

    plt.grid()
    plt.show()

def main(image_directory, number_of_features, ratio_test):
    cv2.destroyAllWindows()  # Close any remaining OpenCV windows

    # Load camera intrinsics
    data = np.load('camera_calib_params.npz')
    K = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

    # Initialize ORB
    orb = cv2.ORB_create(nfeatures=number_of_features)

    # Initialize Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Load images
    try:
        image_files = load_images(image_directory)
    except FileNotFoundError as e:
        print(e)
        return

    # Compute trajectory
    positions, orientations = compute_trajectory(image_files, image_directory, orb, bf, K, dist_coeffs, ratio_test)

    # Smooth the trajectory
    smoothed_positions = smooth_trajectory(positions)

    # Generate ground truth
    ground_truth = generate_ground_truth(image_directory)

    # Plot both trajectories
    plot_trajectories_with_orientations(positions, orientations, smoothed_positions, ground_truth)
