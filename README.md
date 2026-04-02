---
title: KSL Sign Language Translator
emoji: 🤟
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# KSL Sign Language Translator

Real-time sign language translation using MediaPipe and TensorFlow Lite.

## Model Files Required

Place these files in your Space:
- `kls_model_lite.tflite` - Your trained sign language model
- `hand_landmarker.task` - MediaPipe hand landmark model
---
```markdown
# 🤟 KSL Sign Language Translator

Real-time sign language translation application that converts Kenyan Sign Language (KSL) gestures into spoken words using computer vision and machine learning.

## 🌟 Features

- **Real-time Sign Recognition** - Instant translation of sign language gestures
- **Colorful Hand Landmarks** - Visual feedback with color-coded finger tracking
- **Immediate Audio Feedback** - Each recognized sign is spoken aloud immediately
- **Live Video Processing** - Works with your webcam in real-time
- **User-Friendly Interface** - Simple Gradio web interface


## 🎯 Live Demo

Try it live on Hugging Face Spaces: [(https://huggingface.co/spaces/jianna4/KENYASIGNLANGUAGE)]

## 📋 Table of Contents

- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Technical Stack](#technical-stack)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🔍 How It Works

1. **Hand Detection** - MediaPipe detects hand landmarks in real-time
2. **Feature Extraction** - 126 features (63 per hand) extracted from hand positions
3. **Sequence Processing** - Features buffered over 40 frames
4. **Sign Classification** - TensorFlow Lite model predicts the sign
5. **Speech Output** - Recognized signs spoken using text-to-speech

### Visual Guide
- **White Circle** - Palm center
- **Red Thumb** - Thumb tracking
- **Blue Index Finger** - Index finger tracking
- **Green Middle Finger** - Middle finger tracking
- **Yellow Ring Finger** - Ring finger tracking
- **Magenta Pinky** - Pinky finger tracking

##  Installation

### Local Development

1. Clone the repository
```bash
git clone https://github.com/jianna4/Kenya-Sign-Language.git
cd ksl-translator
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download model files
Place your model files in the project directory:
- `kls_model_lite.tflite` - Sign language recognition model
- `hand_landmarker.task` - Hand landmark detection model

5. Run the application
```bash
python app.py
```

### Hugging Face Spaces Deployment

1. Create a new Space on Hugging Face
2. Choose Gradio SDK
3. Upload all project files including:
   - `app.py`
   - `handLandmarks.py`
   - `prediction.py`
   - `texttospeech.py`
   - `requirements.txt`
   - Model files (`.tflite` and `.task`)
4. The Space will automatically build and launch

##  Usage

1. **Allow Camera Access** - Grant permission when prompted
2. **Sign Clearly** - Position your hands in the camera frame
3. **Hold Each Sign** - Maintain the sign for about 1 second
4. **Listen** - The recognized word will be spoken immediately
5. **Reset** - Use the reset button to clear the recognition buffer

### Supported Signs

Currently supports:
- father
- hello
- is
- my

*More signs can be added by retraining the model*

##  Project Structure

```
ksl-translator/
├── app.py                 # Main application and Gradio interface
├── handLandmarks.py       # Hand detection and visualization
├── prediction.py          # TensorFlow Lite model prediction
├── texttospeech.py        # Text-to-speech functionality
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── kls_model_lite.tflite # Sign language model
└── hand_landmarker.task   # Hand landmark model
```

##  Technical Stack

- **Frontend**: Gradio (Web interface)
- **Computer Vision**: MediaPipe, OpenCV
- **Machine Learning**: TensorFlow Lite
- **Speech Synthesis**: gTTS
- **Language**: Python 3.9+

### Key Libraries

```
gradio==4.19.2          # Web interface
opencv-python==4.9.0.80 # Video processing
numpy==1.24.3           # Numerical operations
tensorflow==2.15.0      # Model inference
mediapipe==0.10.9       # Hand landmark detection
pyttsx3==2.90          # Text-to-speech
Pillow==10.1.0         # Image processing
```

## 🛠️ Troubleshooting

### Common Issues

**ModuleNotFoundError**
- Ensure file names match imports exactly (case-sensitive on Linux)
- Check all Python files are uploaded to the Space

**Camera Not Working**
- Verify camera permissions are granted
- Try refreshing the page
- Check if another application is using the camera

**No Signs Detected**
- Ensure good lighting conditions
- Keep hands fully in frame
- Hold signs for at least 1 second
- Wait for colorful landmarks to appear on hands

**TTS Not Working**
- For pyttsx3: May require system audio drivers
- For gTTS: Check internet connection
- Try the alternative TTS implementation

**Slow Performance**
- Reduce MAX_FRAMES in configuration
- Lower video quality in settings
- Close other applications

## 🚀 Future Improvements

- [ ] Support for more signs
- [ ] Sentence-level translation
- [ ] Multiple language support
- [ ] Mobile app version
- [ ] Sign language to text saving
- [ ] Cloud model training interface
- [ ] Custom sign addition feature
- [ ] Performance optimizations for mobile devices

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is licensed under the apache License - see the LICENSE file for details.

##  Acknowledgments

- MediaPipe for hand landmark detection
- TensorFlow for model inference
- Gradio for the web interface
- The KSL community for inspiration

##  Contact

Joan Maina - mainajoan555@gmail.com

Project Link: (https://huggingface.co/spaces/jianna4/KENYASIGNLANGUAGE)

---

**Made with ❤️ for the Kenyan Sign Language community**
```