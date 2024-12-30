import numpy as np

import calibration
import techniques.orb_flann as orb_flann
import techniques.orb_brute as orb_brute
import techniques.sift_flann as sift_flann
import techniques.sift_brute as sift_brute

def main():
    #--------------------------------------------------------------------
    # SET PARAMETERS
    #--------------------------------------------------------------------
    # CALIBRATION PARAMETERS (CHEKBOARD SIZE AND SQUARE SIZE)
    chessboard_size = (8, 6) #8,6
    square_size = 2.5  # 2.5 cm
    
    # PROCESSING PARAMETERS
    number_of_features = 3000 # For ORB 
    ratio_test = 0.75 #Lowe's ratio  test

    #-------------------------------------------------------------------
    # --- SELECT TECHNIQUE (4 choices) ---
    # 'orb_flann' or 'orb_brute'
    # 'sift_flann' or 'sift_brute'
    #--------------------------------------------------------------------
    processing_technique = 'sift_brute'  # Change 'orb_flann' to other choices

    #--------------------------------------------------------------------
    # --- SELECT TRAJECTORY (3 choices) ---
    # 'linear' or 'rectangular' or 'circular'
    #--------------------------------------------------------------------
    trajectory = 'rectangular'  # Change 'linear' to other choices


    #--------------------------------------------------------------------
    # DON'T EDIT BELOW THIS LINE

    #-------------------------------------------------------------------
    #--------------------------------------------------------------------
    # CAMERA CALIBRATION
    #--------------------------------------------------------------------
    if trajectory == 'linear':
        image_directory = 'images/linear/images'
        calibration_directory = 'images/linear/calibration/*.png'

    elif trajectory == 'rectangular':
        image_directory = 'images/rectangular/images'
        calibration_directory = 'images/rectangular/calibration/*.png'

    elif trajectory == 'circular':
        image_directory = 'images/circular/images'
        calibration_directory = 'images/circular/calibration/*.png'
    elif trajectory == 'other':
        image_directory = 'images/other/images'
        calibration_directory = 'images/other/calibration/*.png'
    else:
        print(f"Unknown trajectory: {trajectory}")

    print('\n---------------------------------------')
    print("Calibrating camera... ...\n")

    output_file = 'camera_calib_params.npz'
    
    # CALLING CALIBRATING SCRIPT
    calibration.calibrate_camera(chessboard_size, square_size, calibration_directory, output_file)

    print("Camera calibration completed. \n ")

    #--------------------------------------------------------------------
    # PROCESSING TECHNIQUE
    #--------------------------------------------------------------------
    if processing_technique == 'orb_flann':
        technique_module = orb_flann
    elif processing_technique == 'orb_brute':
        technique_module = orb_brute
    elif processing_technique == 'sift_flann':
        technique_module = sift_flann
    elif processing_technique == 'sift_brute':
        technique_module = sift_brute
    else:
        print(f"Error: Unknown processing technique '{processing_technique}'.")
        return
    
    print(f"Using processing technique: {processing_technique}")
    print(f"Using trajectory: {trajectory}")

    print('\n---------------------------------------')
    print(" \n Estimating camera pose... ...\n")

    # CALLING PROCESSING SCRIPT
    technique_module.main(image_directory,number_of_features,ratio_test)

    print("Camera pose estimation completed. \n ")

if __name__ == "__main__":
    main()
