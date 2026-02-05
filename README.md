 Kenyan Sign Language (KSL) to Speech Translator

EchoSign is an assistive AI system that bridges communication between the Deaf community and the hearing population. It recognizes Kenyan Sign Language (KSL) gestures, converts them into text, and finally transforms that text into speech — enabling real-time communication and inclusivity.

 Overview

EchoSign is built to understand, translate, and vocalize KSL gestures using computer vision and machine learning.

The project’s goal is to empower Deaf and Hard of Hearing individuals by giving them a voice in any conversation through intelligent gesture recognition and speech synthesis.

This version focuses on KSL to Speech, but in future iterations, we will expand to include Speech to KSL, creating a full two-way communication bridge.

 Objectives

Build a Django-based system that detects and translates KSL gestures into English text.

Convert recognized text into audible speech output.

Create a foundation for future Speech-to-KSL translation.

Promote accessibility, inclusivity, and communication equity through AI.

 Project Workflow
STEP 1: Pick 5 High-Impact KSL Words

Select five meaningful and frequently used KSL gestures (e.g., Hello, Thank You, Sorry, Yes, No).
These form the base vocabulary for the initial model.

STEP 2: Record Your Own KSL Vector Dataset (20 mins)

Record short video clips or image frames of each sign.
Ensure:

Consistent lighting and background.

The same signer performs each gesture.

Multiple samples are captured for accuracy.

The dataset will serve as input for the machine learning model.

STEP 3: Train Your KSL Model (5 mins)

Train the model to recognize and classify each KSL gesture.
The model outputs the corresponding English word once a sign is recognized.

This step establishes the foundation of the gesture recognition process.

STEP 4: Run Real-Time KSL → Speech (Demo!)

After training, the model runs in real time:

Capture live video input.

Detect the KSL sign.

Translate it to text.

Convert text to natural speech output.

This final stage demonstrates full translation from gesture to spoken language.

System Architecture
Component Description
Frontend (To Be Added) Will handle camera input and show results.
Backend (Django) Hosts APIs, manages user data, and runs ML models.
ML Engine Recognizes and classifies KSL gestures.
Text-to-Speech Module (TTS) Converts recognized text into spoken audio.
Database (PostgreSQL) Stores gesture data, model results, and logs.
Tech Stack
Layer Technologies
Backend Django, FastAPI (for ML API)
ML/AI TensorFlow / PyTorch
Database PostgreSQL
Frontend To be implemented (React or React Native)
Speech Engine pyttsx3 / gTTS
Version Control Git + GitHub
Deployment Options Railway, Render, or AWS
🧠 How EchoSign Works

The user performs a KSL gesture in front of the camera.

The system captures and processes the image frames.

The trained ML model predicts the corresponding English word or phrase.

The backend converts that text into speech output using a TTS module.

The audio is played, completing gesture-to-speech translation.

🏗️ Project Structure
echosign/
│
├── backend/
│ ├── api/ # Django APIs for ML & TTS
│ ├── models/ # Gesture recognition models
│ ├── tts/ # Text-to-speech handling
│ └── dataset/ # KSL dataset
│
├── requirements.txt
├── manage.py
└── README.md

⚡ Setup & Installation

Clone the Repository
git clone https://github.com/yourusername/echosign.git
cd echosign

Create and Activate a Virtual Environment

Windows: venv\Scripts\activate

Linux/Mac: source venv/bin/activate

Install Dependencies
pip install -r requirements.txt

Configure Database (PostgreSQL)
Update the .env file:

DB_NAME=echosign_db
DB_USER=postgres
DB_PASSWORD=yourpassword
DB_HOST=localhost
DB_PORT=5432

Run Migrations
python manage.py makemigrations
python manage.py migrate

Start the Server
python manage.py runserver

☁️ Deployment

You can deploy EchoSign on:

Railway (simple and free hosting for Django apps)

Render (for production-ready hosting)

AWS EC2 or Lambda (for scalable cloud hosting)

🧭 Future Expansion

✅ Speech-to-KSL Translation – The next phase will enable two-way interaction.
✅ Offline Functionality – For users in low-connectivity regions.
✅ AR/VR Integration – To teach KSL interactively.
✅ Expanded Vocabulary – To include more complex KSL phrases.

👩‍💻 Team

Project Lead: Joan Maina
Domain: Machine Learning | Accessibility AI | Computer Vision
