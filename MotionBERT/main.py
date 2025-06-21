import os
import subprocess
import glob

# Paths
data_folder = "E:\MotionBERT Implementation\Data\Trimmed Video" #Input Trimmed Videos
json_folder = "E:\MotionBERT Implementation\Data\AlphaPose 2D Output" #Input Alpha Pose 2D Json Files
output_folder = "E:\MotionBERT Implementation\Data\MotionBERT 3D Output" #Output of MotionBert

def trim_video_with_ffmpeg(video_path, max_duration=10):
    """
    Trim the video to the specified maximum duration using ffmpeg.
    Overwrites the original video file.
    """
    temp_path = video_path + "_temp.mp4"
    command = [
        "ffmpeg", "-i", video_path, "-t", str(max_duration), "-c", "copy", temp_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(temp_path, video_path)

# Iterate through all MP4 files in the data folder
for video_file in glob.glob(os.path.join(data_folder, "*.mp4")):
    print(f"Processing video: {video_file}")

    # Step 1: Construct the paths for the JSON output
    video_name = os.path.basename(video_file).rsplit('.', 1)[0]  # Get the video name without the extension
    json_path = os.path.join(json_folder, f"{video_name}_2D.json")

    # Step 2: Run the inference script
    script_command = [
        "python",
        "infer_wild.py",
        "--vid_path", video_file,  # Set the video path
        "--json_path", json_path,  # Set the JSON path
        "--out_path", output_folder  # Set the output folder
    ]
    try:
        subprocess.run(script_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the inference script for video: {video_file}")
        continue  # Skip to the next video if the script fails

    # Step 3: Renaming the file
    original_npy_path = os.path.join(output_folder, "X3D.npy")
    new_npy_path = os.path.join(output_folder, f"{video_name}_3D.npy")
    os.rename(original_npy_path, new_npy_path)
    print(f"File renamed to: {new_npy_path}")

print("Processing complete for all videos.")

#     os.rename("examples/alphapose-results.json", f"examples/{os.path.basename(video_file).rsplit('.', 1)[0]}_2D.json")