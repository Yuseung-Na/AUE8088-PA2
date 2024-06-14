import cv2
import glob
from pathlib import Path
from tqdm import tqdm

def main():
    # Directories
    save_dir = Path("./runs/detect/exp")

    # Save result images to video
    img_files = sorted(glob.glob(f"{save_dir}/*.jpg"))  # List of saved image files
    if len(img_files) > 0:
        frame = cv2.imread(img_files[0])
        height, width, layers = frame.shape
        video_path = save_dir / "output_video.mp4"
        video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
        
        for img_file in tqdm(img_files, desc="Saving images to video"):
            frame = cv2.imread(img_file)
            video.write(frame)
        
        video.release()
        print(f"Video saved to {video_path}")

if __name__ == "__main__":
    main()