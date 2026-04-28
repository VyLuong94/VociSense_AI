import os
import sys
import time
import logging
import librosa
import numpy as np
import torch

from torch.nn import functional as F
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# HuggingFace repo
AVAILABLE_MODELS = {
    "dual_emotion": "vyluong/emo_dual_classi"
}

emotion_labels = ['Angry', 'Anxiety', 'Happy', 'Sad', 'Neutral']

EMOTION_META = {
    "Angry": {"emoji": "😡", "color": "#ff4d4f"},
    "Anxiety": {"emoji": "😰", "color": "#faad14"},
    "Happy": {"emoji": "😊", "color": "#52c41a"},
    "Sad": {"emoji": "😢", "color": "#1890ff"},
    "Neutral": {"emoji": "😐", "color": "#d9d9d9"},
}



class EmotionService:

    _models = {}
    
    emotion_labels = emotion_labels
    meta = EMOTION_META

    @classmethod
    def load_dual_model(cls, repo_id, device):

        logger.info(f"Downloading model from HF: {repo_id}")

        model_file = hf_hub_download(
            repo_id=repo_id,
            filename="pytorch_model.bin"
        )

        model_code = hf_hub_download(
            repo_id=repo_id,
            filename="model.py"
        )

        # add model folder to python path
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("hf_model", model_code)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        Dual = module.Dual

        model = Dual()

        state_dict = torch.load(model_file, map_location=device)

        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        logger.info("Emotion model loaded successfully")

        return model



    @classmethod
    def get_model(cls, model_name="dual_emotion"):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name in cls._models:
            return cls._models[model_name]

        repo_id = AVAILABLE_MODELS[model_name]

        model = cls.load_dual_model(repo_id, device)

        cls._models[model_name] = model

        return model


    @classmethod
    def preload_model(cls):

        logger.info("Preloading emotion model...")

        cls.get_model()

        logger.info("Emotion model ready")


    # extract mfcc from segments
    @staticmethod
    def extract_mfcc_segment(
        audio: np.ndarray,
        sr: int,
        start: float,
        end: float,
        duration: float = 5.0,
        n_mfcc: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512
    ):

        start_sample = int(start * sr)
        end_sample = int(end * sr)

        segment = audio[start_sample:end_sample]

        if len(segment) == 0:
            return None

        target_len = int(sr * duration)

        if len(segment) < target_len:
            segment = np.pad(segment,(0,target_len-len(segment)),mode="symmetric")
        else:
            segment = segment[:target_len]

        mfcc = librosa.feature.mfcc(
            y=segment,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )

        return mfcc


    @classmethod
    def predict_from_mfcc(cls, mfcc):

        model = cls.get_model()

        tensor = torch.from_numpy(mfcc).unsqueeze(0).unsqueeze(0).float()

        device = next(model.parameters()).device
        tensor = tensor.to(device)

        with torch.no_grad():

            output = model(tensor)

        probs = F.softmax(output.squeeze(), dim=0).cpu().numpy()

        label = cls.emotion_labels[np.argmax(probs)]

        return label


    # predict from segments
    @classmethod
    def predict_segment(cls, audio, sr, start, end):

        mfcc = cls.extract_mfcc_segment(audio, sr, start, end)

        if mfcc is None:
            return "Neutral"

        return cls.predict_from_mfcc(mfcc)