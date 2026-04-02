"""
Main Gradio application for KSL Sign Language Translator
"""
import gradio as gr
import numpy as np
import cv2
import tempfile
import os
import time
import logging
from PIL import Image
from io import BytesIO
import base64

from handlandmarks import HandLandmarkDetector
from prediction import SignPredictor
from texttospeech import TextToSpeech

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KSLTranslator:
    """Main KSL Translator application"""
    
    def __init__(self, model_path, hand_model_path, labels, 
                 max_frames=40, confidence_threshold=0.7, 
                 prediction_cooldown=1.5):
        """
        Initialize the translator
        
        Args:
            model_path: Path to sign language model
            hand_model_path: Path to hand landmark model
            labels: List of label names
            max_frames: Maximum frames in buffer
            confidence_threshold: Minimum confidence for valid prediction
            prediction_cooldown: Seconds to wait before next prediction
        """
        # Initialize components
        self.hand_detector = HandLandmarkDetector(hand_model_path)
        self.predictor = SignPredictor(model_path, labels, max_frames, confidence_threshold)
        self.tts = TextToSpeech()
        
        # Configuration
        self.max_frames = max_frames
        self.confidence_threshold = confidence_threshold
        self.prediction_cooldown = prediction_cooldown
        
        # State
        self.sequence = []
        self.last_word = ""
        self.full_sentence = []
        self.last_prediction_time = 0
    
    def reset_state(self):
        """Reset all state variables"""
        self.sequence = []
        self.last_word = ""
        self.full_sentence = []
        self.last_prediction_time = 0
        logger.info("State reset")
        return "Conversation reset", "📝 Sentence: ", None
    
    def process_frame(self, frame):
        """
        Process a single video frame
        
        Args:
            frame: Input frame (RGB format)
            
        Returns:
            tuple: (annotated_image, display_text, sentence_text, audio_file, status)
        """
        if frame is None:
            return (None, "No frame", "📝 Sentence: ", None, "Waiting for camera...")
        
        try:
            # Convert RGB to BGR for processing
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                bgr_frame = frame
            
            # Process frame for landmarks
            features, annotated_bgr = self.hand_detector.process_frame(bgr_frame)
            
            # Add to sequence
            self.sequence.append(features)
            if len(self.sequence) > self.max_frames:
                self.sequence = self.sequence[-self.max_frames:]
            
            # Convert back to RGB for display
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            
            # Predict if we have enough frames
            current_word = ""
            confidence = 0.0
            
            if len(self.sequence) >= 10:
                pred_word, conf = self.predictor.predict(self.sequence)
                
                if pred_word and conf >= self.confidence_threshold:
                    current_word = pred_word
                    confidence = conf
                    
                    current_time = time.time()
                    
                    # Check cooldown and avoid duplicates
                    if (current_word != self.last_word or 
                        (current_time - self.last_prediction_time) > self.prediction_cooldown):
                        
                        self.full_sentence.append(current_word)
                        self.last_word = current_word
                        self.last_prediction_time = current_time
                        
                        logger.info(f"Predicted: {current_word} ({confidence:.1%})")
            
            # Prepare display text
            sentence = " ".join(self.full_sentence)
            
            if current_word and confidence >= self.confidence_threshold:
                display_text = f"🧠 **{current_word}** ({confidence:.1%})"
                status_text = f"✅ Recognized: {current_word}"
                sentence_display = f"📝 **Sentence:** {sentence}" if sentence else "📝 **Sentence:** (keep signing...)"
            else:
                display_text = f"👋 Watching... ({len(self.sequence)}/{self.max_frames} frames)"
                status_text = "🤟 Sign a word..."
                sentence_display = f"📝 **Sentence:** {sentence}" if sentence else "📝 **Sentence:** (keep signing...)"
            
            # Generate audio for new word
            audio_file = None
            if current_word and confidence >= self.confidence_threshold:
                try:
                    audio_base64 = self.tts.to_base64(current_word)
                    if audio_base64:
                        audio_bytes = base64.b64decode(audio_base64)
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                        temp_file.write(audio_bytes)
                        temp_file.close()
                        audio_file = temp_file.name
                except Exception as e:
                    logger.error(f"Audio generation error: {e}")
            
            return (annotated_rgb, display_text, sentence_display, audio_file, status_text)
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return (frame, f"⚠️ Error: {str(e)[:100]}", "📝 Sentence: ", None, "Error")
    
    def create_interface(self):
        """Create Gradio interface"""
        
        with gr.Blocks(title="KSL Sign Language Translator", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🤟 KSL Sign Language Translator
            ### Real-time sign language translation with colorful landmarks
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    webcam = gr.Image(
                        label="Live Video Stream",
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        height=520,
                        width=720
                    )
                    
                with gr.Column(scale=1):
                    status = gr.Textbox(
                        label="Status",
                        value="Starting camera...",
                        interactive=False
                    )
                    current_word_box = gr.Markdown("👋 Start signing...")
                    sentence_box = gr.Markdown("📝 Sentence will build here...")
                    audio_output = gr.Audio(
                        label="🔊 Spoken Word",
                        type="filepath",
                        autoplay=True
                    )
                    
                    with gr.Row():
                        reset_btn = gr.Button("🔄 Reset Conversation", variant="secondary")
                        clear_btn = gr.Button("🗑️ Clear Last Word", variant="secondary")
            
            # Reset functionality
            reset_btn.click(
                fn=self.reset_state,
                outputs=[current_word_box, sentence_box, audio_output]
            )
            
            # Clear last word functionality
            def clear_last_word():
                if self.full_sentence:
                    self.full_sentence.pop()
                    self.last_word = ""
                    return "📝 **Sentence:** " + " ".join(self.full_sentence), "✅ Last word removed"
                return "📝 **Sentence:** ", "No words to remove"
            
            clear_btn.click(
                fn=clear_last_word,
                outputs=[sentence_box, status]
            )
            
            # Stream video
            webcam.stream(
                fn=self.process_frame,
                inputs=webcam,
                outputs=[webcam, current_word_box, sentence_box, audio_output, status],
                show_progress=False,
                time_limit=60
            )
            
            # Instructions
            gr.Markdown(f"""
            ### 📖 How to Use:
            1. **Allow camera access** when prompted
            2. **Sign clearly** with hands visible
            3. Each confident sign (≥{self.confidence_threshold*100:.0f}%) will be:
               - Added to the sentence
               - Spoken aloud
            4. Use **Reset** to start a new conversation
            5. Use **Clear Last Word** to remove the most recent word
            
            ### 🎨 Visual Guide:
            - **White circle** = Palm center
            - **Different colors** = Different fingers
            - **Connecting lines** = Finger joints
            
            ### 🏷️ Available Signs:
            {', '.join(self.predictor.labels) if self.predictor.labels else 'Loading...'}
            
            ### 💡 Tips:
            - Ensure good lighting
            - Keep hands fully in frame
            - Hold each sign for about 1 second
            - Wait for green landmarks to appear
            """)
        
        return demo

def main():
    """Main entry point"""
    # Configuration
    MODELS_DIR = os.environ.get('MODELS_DIR', '/app/models')
    HAND_MODEL_PATH = r"F:\projects\KSL\backend2\hand_landmarker.task"
    MODEL_PATH = r"F:\projects\KSL\backend2\kls_model_lite.tflite"

    LABELS = ["father", "hello", "is", "my"]
    
    # Check if models exist
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model not found: {MODEL_PATH}")
        logger.warning("Please ensure the model file exists")
    
    if not os.path.exists(HAND_MODEL_PATH):
        logger.warning(f"Hand model not found: {HAND_MODEL_PATH}")
        logger.warning("Please ensure the hand landmarker file exists")
    
    # Create translator
    translator = KSLTranslator(
        model_path=MODEL_PATH,
        hand_model_path=HAND_MODEL_PATH,
        labels=LABELS,
        max_frames=40,
        confidence_threshold=0.70,
        prediction_cooldown=1.5
    )
    
    # Create and launch interface
    demo = translator.create_interface()
    
    # Launch for Hugging Face Spaces
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    main()