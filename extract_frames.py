import cv2
import os
import glob
import argparse

def extract_frames(video_dir, output_dir, frame_skip=15):
    """
    Extracts frames from all videos in a directory to be used for model training.
    
    :param video_dir: Directory containing the video files (.mp4, .avi, etc.)
    :param output_dir: Directory to save the extracted images
    :param frame_skip: Extract 1 frame every 'frame_skip' frames. 
                       (e.g., if video is 30fps, frame_skip=15 extracts 2 frames per second)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_files = glob.glob(os.path.join(video_dir, '*.[mM][pP]4')) + \
                  glob.glob(os.path.join(video_dir, '*.[aA][vV][iI]')) + \
                  glob.glob(os.path.join(video_dir, '*.[mM][oO][vV]'))

    if not video_files:
        print(f"No videos found in {video_dir}")
        return

    total_extracted = 0

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Failed to open {video_path}")
            continue

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_count:05d}.jpg")
                cv2.imwrite(output_path, frame)
                saved_count += 1
                total_extracted += 1

            frame_count += 1

        cap.release()
        print(f"Extracted {saved_count} frames from {video_name}")

    print(f"\nDone! Extracted a total of {total_extracted} images to {output_dir}")
    print("Next step: Upload these images to an annotation tool like Roboflow or CVAT to label them.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos for ML training.")
    parser.add_argument("--input", type=str, default=".", help="Directory containing videos")
    parser.add_argument("--output", type=str, default="./training_images", help="Output directory for images")
    parser.add_argument("--skip", type=int, default=15, help="Extract 1 frame every N frames")
    
    args = parser.parse_args()
    extract_frames(args.input, args.output, args.skip)
