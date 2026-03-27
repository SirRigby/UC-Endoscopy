import cv2
import os
import time
from pathlib import Path
from tqdm import tqdm

def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def extract_frames(
    video_path,
    output_dir,
    resize=None,
    every_n_frames=1  # new optional param: save every nth frame
):
    start_time = time.time()

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration_sec = total_frames / fps if fps > 0 else 0

    print(f"Total frames in video: {total_frames}")
    print(f"FPS                  : {fps:.2f}")
    print(f"Video length         : {video_duration_sec:.2f} sec ({format_time(video_duration_sec)})")

    frame_idx = 0
    saved_count = 0

    with tqdm(total=total_frames, desc=f"Processing {video_path.name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if resize is not None:
                frame = cv2.resize(frame, resize)

            # save every Nth frame
            if frame_idx % every_n_frames == 0:
                frame_name = f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
                frame_path = output_dir / frame_name
                cv2.imwrite(str(frame_path), frame)
                saved_count += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Video length (sec)     : {video_duration_sec:.2f}")
    print(f"Video length (hh:mm:ss): {format_time(video_duration_sec)}")
    print(f"Total frames           : {total_frames}")
    print(f"Frames processed       : {frame_idx}")
    print(f"Frames saved           : {saved_count}")
    print(f"Time taken (sec)       : {total_time:.2f}")

    if total_time > 0:
        print(f"Processing FPS         : {frame_idx / total_time:.2f}")

    print(f"Saved to {output_dir}")