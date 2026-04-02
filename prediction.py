"""
Sign language prediction using TensorFlow Lite
"""
import numpy as np
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class SignPredictor:
    """Sign language prediction using TFLite model"""
    
    def __init__(self, model_path, labels, max_frames=40, confidence_threshold=0.7):
        """
        Initialize sign predictor
        
        Args:
            model_path: Path to TFLite model
            labels: List of label names
            max_frames: Maximum number of frames to use for prediction
            confidence_threshold: Minimum confidence for valid prediction
        """
        self.model_path = model_path
        self.labels = labels
        self.max_frames = max_frames
        self.confidence_threshold = confidence_threshold
        self.interpreter = None
        self._load_model()
    
    def _load_model(self):
        """Load TFLite model"""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info(f"✅ TFLite model loaded from {self.model_path}")
            logger.info(f"Input shape: {self.input_details[0]['shape']}")
            logger.info(f"Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            self.interpreter = None
    
    def predict(self, sequence):
        """
        Predict sign from feature sequence
        
        Args:
            sequence: List of feature vectors
            
        Returns:
            tuple: (predicted_label, confidence)
        """
        if self.interpreter is None or len(sequence) < 10:
            return "", 0.0
        
        try:
            # Prepare input sequence
            seq_input = np.array(sequence, dtype=np.float32)
            
            # Pad or truncate to max_frames
            if len(seq_input) < self.max_frames:
                pad_size = self.max_frames - len(seq_input)
                seq_input = np.pad(seq_input, ((0, pad_size), (0, 0)), mode='constant')
            elif len(seq_input) > self.max_frames:
                seq_input = seq_input[:self.max_frames]
            
            # Add batch dimension
            seq_input = np.expand_dims(seq_input, axis=0)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], seq_input)
            self.interpreter.invoke()
            preds = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            confidence = float(np.max(preds))
            pred_idx = int(np.argmax(preds))
            
            if (pred_idx < len(self.labels) and 
                confidence >= self.confidence_threshold):
                return self.labels[pred_idx], confidence
            
            return "low confidence", confidence
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return "", 0.0
    
    def is_ready(self):
        """Check if model is loaded"""
        return self.interpreter is not None