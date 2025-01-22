import cv2
import os
import torch
import numpy as np
from PIL import Image
import supervision as sv
from tqdm import tqdm
from utils.video import generate_unique_name, create_directory, delete_directory
from utils.florence import load_florence_model, run_florence_inference, FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from utils.sam import load_sam_image_model, load_sam_video_model, run_sam_inference

class VideoProcessor:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autocast = None

        self.florence_model = None
        self.florence_processor = None
        self.sam_image_model = None
        self.sam_video_model = None

        # Set up mask annotator with a white color palette
        self.mask_annotator = sv.MaskAnnotator(
            color=sv.ColorPalette.from_hex(["#FFFFFF"]),
            color_lookup=sv.ColorLookup.INDEX
        )

    def _load_models(self):
        # Load models
        self.florence_model, self.florence_processor = load_florence_model(device=self.device)
        self.sam_image_model = load_sam_image_model(device=self.device)
        self.sam_video_model = load_sam_video_model(device=self.device)

    def _enable_mixed_precision(self):
        self.autocast = torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        self.autocast.__enter__()
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _reset_mixed_precision(self):
        # Exit the autocast context
        self.autocast.__exit__(None, None, None)
        
        # Reset CUDA settings
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.empty_cache()  # Clear the CUDA cache

    def process_video(self, video_path, scale_factor, prompt):
        self.scale_factor = scale_factor

        # Process video based on the prompt
        output_video_path, session_path, input_frames_dir, output_directory_path = self._process_prompt(video_path, prompt)

        # Create frames from the output video
        fps = self._create_frames(output_video_path, output_directory_path)
        
        # Delete the output video
        os.remove(output_video_path)

        return session_path, fps, input_frames_dir, output_directory_path

    def _create_frames(self, video_path, output_dir):
        create_directory(output_dir)
        # get the video frame width and height
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # open the video file
        cap = cv2.VideoCapture(video_path)

        # Now save all the frames to output_frames folder
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (frame_width, frame_height))
            cv2.imwrite(f"{output_dir}/frame_{count:04d}.jpg", frame)
            count += 1
        return fps

    def _process_prompt(self, video_path, prompt):
        # Process the first frame with the prompt using the loaded models
        frame_generator = sv.get_video_frames_generator(video_path)
        frame = next(frame_generator)
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        texts = [p.strip() for p in prompt.split(",")]
        detections_list = []

        for text in texts:
            _, result = run_florence_inference(
                model=self.florence_model,
                processor=self.florence_processor,
                device=self.device,
                image=frame,
                task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
                text=text
            )
            detections = sv.Detections.from_lmm(
                lmm=sv.LMM.FLORENCE_2,
                result=result,
                resolution_wh=frame.size
            )
            detections = run_sam_inference(self.sam_image_model, frame, detections)
            detections_list.append(detections)

        # Merge detections from all prompts
        detections = sv.Detections.merge(detections_list)
        detections = run_sam_inference(self.sam_image_model, frame, detections)

        # Check if any objects were detected
        if len(detections.mask) == 0:
            raise ValueError(f"No objects of class {', '.join(texts)} found in the first frame of the video.")

        # Generate unique name for video processing
        name = generate_unique_name()
        # session_path = os.path.join("tmp", name)
        # create_directory(session_path)
        # frame_directory_path = os.path.join(session_path, "input_frames")
        # create_directory(frame_directory_path)
        import tempfile
        session_path = tempfile.mkdtemp(prefix="video_processing_")
        frame_directory_path = tempfile.mkdtemp(prefix="input_frames_", dir=session_path)
        output_directory_path = tempfile.mkdtemp(prefix="output_frames_", dir=session_path)

        frames_sink = sv.ImageSink(
            target_dir_path=frame_directory_path,
            image_name_pattern="{:05d}.jpeg"
        )

        # Get video info and scale
        video_info = sv.VideoInfo.from_video_path(video_path)
        video_info.width = int(video_info.width * self.scale_factor)
        video_info.height = int(video_info.height * self.scale_factor)

        # Split video into frames
        frames_generator = sv.get_video_frames_generator(video_path)
        with frames_sink:
            for frame in tqdm(frames_generator, total=video_info.total_frames, desc="Splitting video into frames"):
                frame = sv.scale_image(frame, self.scale_factor)
                frames_sink.save_image(frame)

        # Initialize SAM video model state
        inference_state = self.sam_video_model.init_state(
            video_path=frame_directory_path,
            device=self.device
        )

        # Add masks to inference state
        for mask_index, mask in enumerate(detections.mask):
            _, _, _ = self.sam_video_model.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=mask_index,
                mask=mask
            )

        # Create output video path
        # output_video_path = os.path.join("tmp", f"{name}.mp4")
        output_video_path = os.path.join(session_path, f"{name}.mp4")
        frames_generator = sv.get_video_frames_generator(video_path)
        masks_generator = self.sam_video_model.propagate_in_video(inference_state)

        # Process and annotate each frame
        with sv.VideoSink(output_video_path, video_info=video_info) as sink:
            for frame, (_, tracker_ids, mask_logits) in zip(frames_generator, masks_generator):
                frame = sv.scale_image(frame, self.scale_factor)
                masks = (mask_logits > 0.0).cpu().numpy().astype(bool)
                if len(masks.shape) == 4:
                    masks = np.squeeze(masks, axis=1)

                detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks=masks),
                    mask=masks,
                    class_id=np.array(tracker_ids)
                )

                annotated_frame = frame.copy()

                annotated_frame[:, :, :] = 0
                
                annotated_frame = self.mask_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )
                annotated_frame = (annotated_frame > 0).astype(np.uint8) * 255
                sink.write_frame(annotated_frame)

        return output_video_path, session_path, frame_directory_path, output_directory_path

#Example usage
# output_video = video_processor.process_video(
#     video_path="videos/clip-07-camera-2.mp4", 
#     scale_factor=0.5, 
#     prompt="players, basketball, rim, players shadow"
# )
# print(f"Processed video saved at: {output_video}")