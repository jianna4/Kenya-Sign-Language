"""
Text to Speech module using gTTS
"""
import io
import base64
import logging
from gtts import gTTS

logger = logging.getLogger(__name__)

class TextToSpeech:
    """Convert text to speech and return as base64 audio"""
    
    def __init__(self, lang='en', slow=False):
        """
        Initialize TTS
        
        Args:
            lang: Language code (default: 'en')
            slow: Whether to speak slowly (default: False)
        """
        self.lang = lang
        self.slow = slow
    
    def to_base64(self, text):
        """
        Convert text to speech and return as base64 string
        
        Args:
            text: Text to convert to speech
            
        Returns:
            base64 encoded audio string, or None if error
        """
        if not text or not text.strip():
            return None
        
        try:
            tts = gTTS(text=text, lang=self.lang, slow=self.slow)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    def save_to_file(self, text, filepath):
        """
        Save speech to file
        
        Args:
            text: Text to convert
            filepath: Path to save audio file
            
        Returns:
            bool: Success status
        """
        try:
            tts = gTTS(text=text, lang=self.lang, slow=self.slow)
            tts.save(filepath)
            return True
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False