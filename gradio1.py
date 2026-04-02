import gradio as gr
import requests
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import time

BACKEND_URL = "http://127.0.0.1:8000/predict"
previous_prediction = ""
last_sentence = ""

def process_frame(frame, state):
    global previous_prediction, last_sentence
    
    if frame is None:
        return None, "No frame", "📝 Sentence: ", None, "Waiting for camera..."
    
    try:
        # Convert frame to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            if frame.dtype == np.uint8:
                # Ensure it's BGR for sending to backend
                if frame[0,0,0] > frame[0,0,2]:  # Simple check if it's already BGR
                    bgr_frame = frame
                else:
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                bgr_frame = (frame * 255).astype(np.uint8)
                bgr_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_RGB2BGR)
        else:
            bgr_frame = frame
        
        # Encode frame
        _, buffer = cv2.imencode(".jpg", bgr_frame)
        files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
        
        # Send to backend
        response = requests.post(BACKEND_URL, files=files, timeout=5)
        
        if response.status_code != 200:
            return (
                Image.fromarray(frame),
                f"❌ Backend error: {response.status_code}",
                last_sentence,
                None,
                "Backend error"
            )
        
        result = response.json()
        
        # Extract data
        current_word = result.get("prediction", "")
        confidence = result.get("confidence", 0.0)
        sentence = result.get("sentence", "")
        audio_base64 = result.get("audio")
        annotated_base64 = result.get("annotated_frame")
        frames_collected = result.get("frames", 0)
        error = result.get("error", "")
        
        if error:
            return (
                Image.fromarray(frame),
                f"⚠️ {error}",
                last_sentence,
                None,
                "Error"
            )
        
        # Update last_sentence if changed
        if sentence != last_sentence:
            last_sentence = sentence
        
        # Decode annotated frame
        try:
            if annotated_base64 and "base64" in annotated_base64:
                img_data = annotated_base64.split(",")[1]
                img_bytes = base64.b64decode(img_data)
                annotated_pil = Image.open(BytesIO(img_bytes))
            else:
                annotated_pil = Image.fromarray(frame)
        except:
            annotated_pil = Image.fromarray(frame)
        
        # Display text
        if current_word and confidence >= 0.70:
            display_text = f"🧠 **{current_word}** ({confidence:.1%})"
            status_text = f"✅ Recognized: {current_word}"
        else:
            display_text = f"👋 Watching... ({frames_collected}/40 frames)"
            status_text = "🤟 Sign a word..."
        
        sentence_display = f"📝 **Sentence:** {sentence}" if sentence else "📝 **Sentence:** (keep signing...)"
        
        # Handle audio
        audio_to_play = None
        if audio_base64 and current_word and current_word != previous_prediction:
            try:
                # Convert base64 to audio bytes
                audio_bytes = base64.b64decode(audio_base64)
                # Save to temporary file for Gradio
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_file.write(audio_bytes)
                temp_file.close()
                audio_to_play = temp_file.name
                previous_prediction = current_word
                status_text = f"🔊 Speaking: {current_word}"
            except Exception as e:
                print(f"Audio error: {e}")
                audio_to_play = None
        
        return (
            annotated_pil,
            display_text,
            sentence_display,
            audio_to_play,
            status_text
        )
        
    except requests.exceptions.ConnectionError:
        return (
            Image.fromarray(frame) if isinstance(frame, np.ndarray) else None,
            "❌ Backend not running",
            "Please start the backend server",
            None,
            "Start FastAPI first: uvicorn main:app --reload"
        )
    except Exception as e:
        print(f"Processing error: {e}")
        return (
            Image.fromarray(frame) if isinstance(frame, np.ndarray) else None,
            f"⚠️ Error: {str(e)[:100]}",
            last_sentence,
            None,
            "Processing error"
        )

# Create Gradio interface
with gr.Blocks(title="KSL Live Translator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤟 KSL Sign Language Translator")
    gr.Markdown("**Continuous Live Video** → Text + Speech with landmarks")
    
    with gr.Row():
        with gr.Column(scale=2):
            webcam = gr.Image(
                label="Live Video Stream (green landmarks = hands detected)",
                sources=["webcam"],
                streaming=True,
                type="numpy",
                height=520,
                width=720,
                interactive=True
            )
            
        with gr.Column(scale=1):
            status = gr.Textbox(label="Status", value="Starting camera...", interactive=False)
            current_word_box = gr.Markdown("👋 Start signing...")
            sentence_box = gr.Markdown("📝 Sentence will build here...")
            audio_output = gr.Audio(
                label="🔊 Spoken Word",
                type="filepath",
                autoplay=True,
                interactive=False
            )
            
            with gr.Row():
                reset_btn = gr.Button("🔄 Reset Conversation")
    
    # Reset function
    def reset_conversation():
        global previous_prediction, last_sentence
        previous_prediction = ""
        last_sentence = ""
        try:
            requests.get("http://127.0.0.1:8000/reset")
        except:
            pass
        return "👋 Conversation reset", "📝 Sentence: ", None
    
    reset_btn.click(
        fn=reset_conversation,
        outputs=[current_word_box, sentence_box, audio_output]
    )
    
    # Stream video
    webcam.stream(
        fn=process_frame,
        inputs=[webcam, gr.State(value=None)],
        outputs=[webcam, current_word_box, sentence_box, audio_output, status],
        show_progress=False,
        time_limit=60
    )
    
    gr.Markdown("""
    ### How to use:
    1. **Start the backend** first: `uvicorn main:app --reload`
    2. **Launch this interface** (it will open automatically)
    3. **Allow camera access** when prompted
    4. **Sign clearly** with hands in view
    5. Each confident word (≥70%) will be:
       - Added to the sentence
       - Spoken aloud
    6. Use **Reset** button to start a new conversation
    
    ### Troubleshooting:
    - Make sure your hands are well-lit and visible
    - Wait for green landmarks to appear on hands
    - Each sign needs to be held for ~1 second
    - Check backend console for prediction logs
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True
    )