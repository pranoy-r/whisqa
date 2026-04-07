import math
import torch
import torchaudio

from whisper_score.models.whisper_ni_predictors import whisperMetricPredictorEncoderLayersTransformerSmall, whisperMetricPredictorEncoderLayersTransformerSmalldim
import sys
import torchaudio
import argparse
import torch
import os
from importlib import resources


def get_score(audio_file: str, model_type: str) -> torch.Tensor:
    """
    Get a score for a given audio file using chunking for long files.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SINGLE_MODEL_URL = "https://github.com/pranoy-r/whisqa/releases/download/0.0.1/single_head_model.pt"
    MULTI_MODEL_URL = "https://github.com/pranoy-r/whisqa/releases/download/0.0.1/multi_head_model.pt"

    # --- 1. Load Model ---
    if model_type == "single":
        model = whisperMetricPredictorEncoderLayersTransformerSmall()
        state_dict = torch.hub.load_state_dict_from_url(SINGLE_MODEL_URL, map_location=device)
        model.load_state_dict(state_dict)
    elif model_type == "multi":
        model = whisperMetricPredictorEncoderLayersTransformerSmalldim()
        state_dict = torch.hub.load_state_dict_from_url(MULTI_MODEL_URL, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise ValueError("Model type not supported")

    model.eval()
    model.to(device)

    # --- 2. Load & Format Audio ---
    waveform, sample_rate = torchaudio.load(audio_file)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # --- 3. Chunking Logic ---
    # Whisper's limit is 30 seconds (480,000 samples at 16kHz)
    chunk_size = 16000 * 30 
    total_samples = waveform.shape[1]
    
    # If audio is 30s or shorter, process it normally
    if total_samples <= chunk_size:
        waveform = waveform.to(device)
        with torch.no_grad():
            score = model(waveform)
            if model_type == "multi":
                score = score.squeeze(0)
        return score

    # If audio is longer than 30s, split it into chunks
    scores = []
    num_chunks = math.ceil(total_samples / chunk_size)
    
    with torch.no_grad():
        for i in range(num_chunks):
            start_sample = i * chunk_size
            end_sample = min((i + 1) * chunk_size, total_samples)
            chunk = waveform[:, start_sample:end_sample]
            
            # Skip trailing chunks that are less than 2 seconds long
            # (Extremely short clips can sometimes confuse the transformer and skew the average)
            if chunk.shape[1] < (16000 * 2) and i > 0:
                continue
                
            chunk = chunk.to(device)
            chunk_score = model(chunk)
            
            if model_type == "multi":
                chunk_score = chunk_score.squeeze(0)
                
            scores.append(chunk_score)
            
    # --- 4. Average the Results ---
    # Stack the tensor scores and calculate the mean across the chunks
    stacked_scores = torch.stack(scores)
    final_averaged_score = torch.mean(stacked_scores, dim=0)
    
    return final_averaged_score