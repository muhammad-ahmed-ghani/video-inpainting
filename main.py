import os
import shutil
import torch
from propainter_pipeline import process_video
from florancesam_pipeline import VideoProcessor
import tempfile

video_processor = VideoProcessor()
video_processor._enable_mixed_precision()
video_processor._load_models()

def infer(video_path, scale_factor, prompt):
    video_processor._enable_mixed_precision()
    # --------------------------------------------------------------------------------
    print("Processing video with FlorenceSam...")
    session_path, fps, input_frames_dir, output_frames_dir = video_processor.process_video(
        video_path=video_path,
        scale_factor=scale_factor,
        prompt=prompt
    )
    print(f"Processed video saved at: {session_path}") 
    print(f"FPS: {fps}")
    # --------------------------------------------------------------------------------

    print("Cleaning up...")
    video_processor._reset_mixed_precision()
    torch.cuda.empty_cache()

    print("Processing video with ProPainter...")
    result_path = tempfile.mkdtemp()
    inpainted_video = process_video(video=input_frames_dir, mask=output_frames_dir, save_fps=int(fps), fp16=True, output=result_path)

    shutil.rmtree(input_frames_dir)
    shutil.rmtree(output_frames_dir)
    torch.cuda.empty_cache()
    
    return inpainted_video

if __name__ == "__main__":
    infer("/home/ubuntu/ahmedghani/clip-07-camera-2.mp4", 0.5, "players, basketball, rim, players shadow")