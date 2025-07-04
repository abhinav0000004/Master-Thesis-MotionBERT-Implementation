
# 3D Pose Estimation Using AlphaPose and MotionBERT

This repository contains a setup for 3D pose estimation integrating two powerful models: AlphaPose for 2D pose estimation and MotionBERT for 3D pose estimation. Below are the guidelines for setting up and running the project.

## Prerequisites

Before running the code, make sure to set up the environment by following these guides:

- [AlphaPose GitHub Repository](https://github.com/MVIG-SJTU/AlphaPose/tree/master)
- [MotionBERT GitHub Repository](https://github.com/Walter0807/MotionBERT/tree/main)
- [Additional AlphaPose Setup Instructions](https://blog.csdn.net/weixin_44848751/article/details/132140935)

Ensure that all necessary libraries are installed and that the model weights are correctly placed in their respective directories as detailed in the GitHub repositories.

**Note:** Use the code from this repo as it has been significantly modified from the original versions available on GitHub. This ensures compatibility and correct functioning of the integrated system. And check the original MotionBERT and AlphaPose Repo for folder structures and placing the weights and checkpoints

## Project Structure

- **AlphaPose Folder**: Contains the modified AlphaPose code from the [AlphaPose GitHub](https://github.com/MVIG-SJTU/AlphaPose/tree/master).
- **MotionBERT Folder**: Contains the MotionBERT code from the [MotionBERT GitHub](https://github.com/Walter0807/MotionBERT/tree/main).
- **Data**: This directory includes several subfolders:
  - **Original Video**: Place your videos here.
  - **Ground Truth**: Place ground truth JSON files here.
  - **AlphaPose 2D Output**: Stores the 2D JSON output generated by AlphaPose.
  - **Trimmed Videos**: Stores the first 10 seconds of the original videos, trimmed.
  - **MotionBERT 3D Output**: Stores the 3D NPY files generated by MotionBERT.
  - **Ground Truth Calculated Angles**: Excel files containing ground truth angles for each exercise and consolidated.
  - **MotionBERT Calculated Angles**: Excel files containing angles calculated by MotionBERT for each exercise and consolidated.
 - **Calculate_Angle.py**: This script calculates angles from 3D JSON/NPY files.
 - **Calculate_MPJAE.py**: This script calculates MPJAE between the  ground truth angles and predicted angles.

## Running the Code

1. **Set Up Environment**: Follow the above guides to install the necessary libraries and place model weights.
2. **Prepare Data**: Add your videos and JSON ground truth files to `Data/original Video` and `Data/Ground Truth` respectively.
3. **Run AlphaPose**:
   - Navigate to the AlphaPose folder.
   - Modify input-output directories in `main.py`.
   - Execute `main.py`. It processes the first 10 seconds (approximately 503 frames) of each video, outputting 2D JSON files to `Data/AlphaPose 2D Output`.
4. **Run MotionBERT**:
   - Navigate to the MotionBERT folder.
   - Adjust directories in the main script.
   - Execute the script to generate 3D pose output (NPY files) from 2D JSON outputs, stored in `Data/MotionBERT 3D Output`.
5. **Calculate Angles**:
   - Execute `Calculate_Angles.py` to compute various angles.
   - The results are saved directly in the Data folder.
   - Individual Action Files: For each action, an individual file is created that contains the computed angles.
   - Consolidated Files: Two consolidated files are created:
   - Ground_Truth_Angles_Processed/MotionBERT_Angles_Processed: Contains data processed from MotionBERT's output, where the first frame's data has been subtracted from the rest of the frames in each action sequence to reduce initial bias.
   - Ground_Truth_Angles/MotionBERT_Angles: Absolute Angles.
     
6. **Evaluate MPJAE**:
   - Run `Calculate_MPJAE.py` to calculate the Mean Per Joint Angle Error (MPJAE) between ground truth and MotionBERT's predicted results. You can choose whether you want to use processed angles or absolute angles and modify the code accordingly (For thesis I used processed angles). The output will be stored in the Data folder in Excel format.



   This project uses code and models from the following repositories:

- [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) — Licensed under [Apache License 2.0](https://github.com/MVIG-SJTU/AlphaPose/blob/master/LICENSE).
- [MotionBERT](https://github.com/Walter0807/MotionBERT) — Licensed under [CC BY-NC-SA 4.0](https://github.com/Walter0807/MotionBERT/blob/main/LICENSE).

All rights and acknowledgments belong to the respective authors.
