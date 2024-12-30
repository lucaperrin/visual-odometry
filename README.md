# VISUAL ODOMETRY - "TOURNER AUTOUR DU POT"

## Overview

This project implements a monocular visual odometry pipeline to estimate camera motion around a fixed object with textured surfaces. The project supports different trajectories (linear, rectangular, circular).

The project is titled "Tourner Autour du Pot", as it focuses on analyzing camera movements around a central object, such as a box with patterns on its faces.

## Objective

The goal is to:

- Perform camera calibration to estimate intrinsic parameters.
- Estimate camera trajectories by processing a sequence of images.
- Compare the estimated trajectory with the ground truth.
- Visualize and analyze results with various trajectory shapes.

## How It Works

### Main Components

**Calibration:**

- `chessboard_size`: The number of inner squares on the chessboard used for camera calibration.
- `square_size`: The size of one square on the chessboard, in centimeters.

**Processing Parameters:**

- `number_of_features`: The number of keypoints detected in each image.
- `ratio_test`: The threshold for Lowe's ratio test, used to filter poor matches during feature matching.

**Technique Selection:**

- `processing_technique`: Select from one of the following feature matching methods:
    - `orb_flann`
    - `orb_brute`
    - `sift_flann`
    - `sift_brute`

**Trajectory Analysis:**

- `trajectory`: Choose the trajectory shape to analyze:
    - `linear`
    - `rectangular`
    - `circular`

## What's Included

### Code Files

- `main.py`: Entry point of the project, where you can set parameters, process trajectories, and visualize results.
- `calibration.py`: Script for camera calibration using a chessboard pattern.
- `camera_calib_params.npz`: Precomputed camera calibration parameters (intrinsic matrix and distortion coefficients).
- `techniques/`: Directory containing implementations of feature detection and matching techniques:
    - `orb_flann.py`: ORB + FLANN-based matching.
    - `orb_brute.py`: ORB + brute-force matching.
    - `sift_flann.py`: SIFT + FLANN-based matching.
    - `sift_brute.py`: SIFT + brute-force matching.

### Image Directories

Each trajectory directory contains:

- `images/`: Raw images used for trajectory estimation.
- `calibration/`: Calibration images used for chessboard-based camera calibration.

## Structure

```
images/
├── linear/
│   ├── images/
│   └── calibration/
├── rectangular/
│   ├── images/
│   └── calibration/
└── circular/
        ├── images/
        └── calibration/
```

## Plots

Output plots comparing the estimated trajectory, smoothed trajectory, and ground truth trajectory.

## Usage

### Setup

Clone the repository:

```sh
git clone <lucaperrin/visual-odometry>
cd <repository-directory>
```

### Visualization

The script generates visualizations of:

- Keypoint matching between image pairs.
- The estimated trajectory with orientations.
- Comparison with ground truth.

## Features

- **Camera Calibration**: Ensures accurate pose estimation using chessboard calibration images.
- **Flexible Processing Techniques**: Supports ORB and SIFT-based methods with FLANN or brute-force matchers.
- **Trajectory Analysis**: Choose from predefined trajectory types (linear, rectangular, circular).
- **Outlier Removal and Smoothing**: Includes RANSAC filtering, outlier removal, and Kalman smoothing for robust trajectory estimation.
- **Ground Truth Comparison**: Compare estimated trajectories with known ground truth.

## Results

- Estimated trajectory plotted alongside smoothed and ground truth trajectories.
- Visualizations of keypoints and matches for each image pair.