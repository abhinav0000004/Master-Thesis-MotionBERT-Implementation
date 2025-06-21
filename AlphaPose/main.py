import os
import subprocess
import glob
import shutil
import argparse
import json

# # Paths
data_folder = r"E:\MotionBERT Implementation\Data\Original Video" #Folder to pick original Videos From
trimmed_videos_folder = r"E:\MotionBERT Implementation\Data\Trimmed Video" #Folder to store trimmed videos
json_output_folder= r"E:\MotionBERT Implementation\Data\AlphaPose 2D Output" # Folder to Store 2D JSON Output
examples_folder = "examples" 
demo_folder = os.path.join(examples_folder, "demo")  #Folder where images will be stored temporarily

# Ensure the necessary folders exist
os.makedirs(demo_folder, exist_ok=True)
os.makedirs(trimmed_videos_folder, exist_ok=True)

script_command = [
    "python", 
    "scripts/demo_inference.py", 
    "--cfg", "configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml", 
    "--checkpoint", "pretrained_models/halpe26_fast_res50_256x192.pth", 
    "--indir", demo_folder
#    "--save_img"
]

def trim_video_with_ffmpeg(video_path, max_duration=10):
    """
    Trim the video to the specified maximum duration using ffmpeg.
    Stores the trimmed video in a separate directory.
    """
    base_name = os.path.basename(video_path)
    output_path = os.path.join(trimmed_videos_folder, base_name)
    command = [
        "ffmpeg", "-i", video_path, "-t", str(max_duration), "-c", "copy", output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

# Function to extract images from the video
def extract_images(video_path, output_folder):
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, os.path.join(output_folder, "%04d.jpg")],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError as e:
        print(f"Error extracting images from video: {video_path}")
        return False
    return True

def remove_duplicate_json_entries(json_file_path):
    """
    Reads a JSON file, removes duplicate image_id entries (keeping only the first occurrence),
    and overwrites the cleaned data back into the original JSON file.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: File '{json_file_path}' not found.")
        return

    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        if not isinstance(data, list):
            print(f"Invalid JSON format in file: {json_file_path}. Expected a list.")
            return

        seen_images = set()
        unique_data = []

        for entry in data:
            image_id = entry.get("image_id")
            if image_id and image_id not in seen_images:
                seen_images.add(image_id)
                unique_data.append(entry)

        # Overwrite the original file with cleaned data
        with open(json_file_path, 'w') as file:
            json.dump(unique_data, file, indent=4)

        print(f"Removed duplicates from: {json_file_path}")

    except json.JSONDecodeError:
        print(f"Invalid JSON format in file: {json_file_path}")
    except Exception as e:
        print(f"Error processing file {json_file_path}: {str(e)}")


# Iterate through all MP4 files in the data folder
for video_file in glob.glob(os.path.join(data_folder, "*.mp4")):
    print(f"Processing video: {video_file}")

    # Step 1: Trim the video to 10 seconds
    trimmed_video_path = trim_video_with_ffmpeg(video_file)

    # Step 2: Extract images from the trimmed video
    if not extract_images(trimmed_video_path, demo_folder):
        continue  # Skip this file if image extraction fails

    # Step 3: Run the inference script
    try:
        subprocess.run(script_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the script for video: {trimmed_video_path}")
        continue  # Skip cleanup if script fails

    #Step 3.1: Rename the JSON output file
    video_name = os.path.basename(trimmed_video_path).rsplit('.', 1)[0]  # Get the video name without the extension
    source_file_path = "examples/alphapose-results.json"
    destination_file_path = os.path.join(json_output_folder, f"{video_name}_2D.json")
    shutil.move(source_file_path, destination_file_path)
    print(f"Moved JSON file to: {destination_file_path}")

    # Step 4: Clean up - Delete all images in the demo folder
    for file in os.listdir(demo_folder):
        file_path = os.path.join(demo_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Step 5: Check and remove duplicates from the JSON file
    remove_duplicate_json_entries(destination_file_path)

print("Processing complete for all videos.")
