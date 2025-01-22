from main import infer
import moviepy.editor as mp
import gradio as gr
import os
import tempfile

def pre_processor(video_path, scale_factor, prompt, crop_duration):
    video = mp.VideoFileClip(video_path)
    cropped_video = video.subclip(0, min(crop_duration, video.duration))

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_output = temp_file.name
        cropped_video.write_videofile(temp_output, codec="libx264")

        output = infer(temp_output, scale_factor, prompt)

        # Clean up temporary file
        os.unlink(temp_output)

    return output

demo = gr.Interface(
    title="Text Based Video Inpainting 🔥 (SAM2+Florance2+ProPainter)",
    fn=pre_processor, 
    inputs=[
        gr.Video(label="Upload Video"), 
        gr.Slider(0.1, 1.0, step=0.1, value=0.5, label="Resize Scale Factor (Due to OOM error)"), 
        gr.Textbox(label="Prompt"), 
        gr.Slider(1, 12, step=1, value=6, label="Crop Duration (seconds)")
    ], 
    outputs="video"
)

demo.launch(server_port=7555)
