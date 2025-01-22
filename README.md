# Text-Based Video Inpainting 🎥
This project combines SAM2, Florence2, and ProPainter for intelligent video object removal using text prompts. Simply describe which objects you want to remove (e.g "man, car, cap, basket"), and the AI will do the rest! 🪄

## ✨ Features

- 🎯 Text-guided object removal from videos
- 📹 Support for common video formats
- 🔄 Adjustable scale factor for memory optimization
- ⚡ GPU-accelerated processing
- ⏱️ Customizable processing duration

## 🚀 Quick Demo

### Try it now on Hugging Face Spaces:
  <a href="https://huggingface.co/spaces/ahmedghani/video-object-removal"><img src="https://img.shields.io/static/v1?label=Demo&message=HuggingFace&color=yellow"></a>

## 💻 Installation:
```bash
conda create -n video-inpainting python=3.11 -y
conda activate video-inpainting
pip install -r requirements.txt
```

## 📥 Download Checkpoints

To get started with the project, you'll need to download the required pretrained models and organize them as follows:

Download the SAM2 checkpoints from here:  **[SAM2_PRETRAINED_MODELS](https://huggingface.co/ahmedghani/video-inpainting-checkpoints/tree/main/checkpoints)**  
🗂️ Place the downloaded files in the following directory: `./checkpoints`

---

Download the ProPainter checkpoints from here:  **[PROPAINTER_PRETRAINED_MODELS](https://huggingface.co/ahmedghani/video-inpainting-checkpoints/tree/main/weights)**  
🗂️ Place the downloaded files in the following directory:  `./weights`

## 🎮 Usage
Run the Gradio interface:
```bash
python app.py
```

### 🔧 System Requirements:
- CUDA-capable GPU with 16GB+ VRAM
- CUDA 12.1 or higher
- 16GB+ RAM
- Python 3.11
