#### MODEL LOADING CODE################
import torch

import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
import torchaudio
import torchvision
import numpy as np
import os
import cv2
from PIL import Image
import librosa
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== TEXT PROCESSING MODULE ====================
class TextEncoder(nn.Module):
    def __init__(self, pretrained_model="distilbert-base-uncased", hidden_size=768):
        super(TextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModel.from_pretrained(pretrained_model).to(device)

        # ❄️ Freeze BERT
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.fc = nn.Linear(hidden_size, 256).to(device)
        
    def forward(self, texts):
        # Tokenize texts
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Get text embeddings
        outputs = self.model(**inputs)
        text_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        text_features = self.fc(text_features)
        return text_features



# ==================== AUDIO ENCODER USING HUBERT ====================
import torchaudio
from torchaudio.pipelines import HUBERT_BASE
from torchaudio.transforms import Resample

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):  # input_dim kept for compatibility
        super(AudioEncoder, self).__init__()

        # Load HuBERT base model from torchaudio
        self.hubert_bundle = HUBERT_BASE
        self.hubert = self.hubert_bundle.get_model().to(device)

        # Freeze HuBERT parameters (can be unfrozen later for fine-tuning)
        for param in self.hubert.parameters():
            param.requires_grad = False

        # Projection layer: HuBERT output (768-dim) -> hidden_dim
        self.project = nn.Sequential(
            nn.Linear(self.hubert_bundle._params['encoder_embed_dim'], 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, hidden_dim)
        )

        self.to(device)

    def forward(self, waveforms):
        """
        Args:
            waveforms: Tensor [B, T] or [B, 1, T] (mono)
        Returns:
            Tensor: [B, hidden_dim]
        """
        # Ensure waveforms are [B, T]
        if waveforms.dim() == 3 and waveforms.shape[1] == 1:
            waveforms = waveforms.squeeze(1)

        with torch.no_grad():
            features, _ = self.hubert(waveforms)  # [B, T', 768]
            pooled = features.mean(dim=1)         # [B, 768]

        return self.project(pooled)               # [B, hidden_dim]

    @staticmethod
    def extract_spectrogram(audio_path, target_sr=16000, fixed_len=16000):
        """
        Preprocess .wav audio file into fixed-length waveform tensor.
        Args:
            audio_path (str): Path to a .wav file
            target_sr (int): Target sampling rate
            fixed_len (int): Desired number of samples (default: 16000 = 1 second)
        Returns:
            waveform (Tensor): [1, fixed_len], float32, 16kHz mono
        """
        waveform, sr = torchaudio.load(audio_path)
    
        # Mono conversion
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
    
        # Resample to 16kHz
        if sr != target_sr:
            resample = Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resample(waveform)
    
        # Pad or truncate to fixed_len
        num_samples = waveform.shape[1]
        if num_samples < fixed_len:
            pad_size = fixed_len - num_samples
            waveform = F.pad(waveform, (0, pad_size))
        elif num_samples > fixed_len:
            waveform = waveform[:, :fixed_len]
    
        return waveform.to(device)



import timm
import torch.nn as nn
import torch.nn.functional as F

class VideoEncoder(nn.Module):
    def __init__(self, hidden_dim=256):
        super(VideoEncoder, self).__init__()
        
        # Load pretrained Xception backbone (no classifier)
        self.backbone = timm.create_model('xception', pretrained=True, num_classes=0).to(device)
        
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Projection layer
        self.fc = nn.Linear(2048, hidden_dim)
        self.to(device)

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)  # [B, 1, 3, H, W] → [B, 3, H, W]
        features = self.backbone(x)           # [B, 2048]
        return self.fc(features)              # [B, hidden_dim]
    
    @staticmethod
    def extract_face_features(video_file, num_frames=16):
        """Extract facial features from video frames"""
        # Initialize face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Open video file
        cap = cv2.VideoCapture(video_file)
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices to extract
        indices = np.linspace(0, frame_count-1, num_frames, dtype=int)
        
        # Initialize tensor to store face frames
        face_frames = torch.zeros((num_frames, 3, 224, 224), device=device)

        
        for i, idx in enumerate(indices):
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Extract the largest face
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                
                # Expand bounding box slightly
                x = max(0, x - int(0.1 * w))
                y = max(0, y - int(0.1 * h))
                w = min(frame.shape[1] - x, int(1.2 * w))
                h = min(frame.shape[0] - y, int(1.2 * h))
                
                # Extract face
                face = frame[y:y+h, x:x+w]
                
                # Resize to 224x224
                face = cv2.resize(face, (224, 224))
                
                # Convert to RGB
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                
                # Convert to tensor
                face_tensor = torchvision.transforms.ToTensor()(face)
                
                # Normalize
                face_tensor = torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )(face_tensor).to(device)
                
                # Store tensor
                face_frames[i] = face_tensor
        
        # Release video capture
        cap.release()
        
        # Return the mean face features across frames with correct shape
        # This will ensure shape is [1, 3, 224, 224]
        return face_frames.mean(dim=0).unsqueeze(0)



# ==================== ATTENTION FUSION MODULE ====================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        
    def forward(self, query, key, value):
        # For PyTorch's MultiheadAttention, inputs should be: [seq_len, batch_size, embed_dim]
        # Make sure all inputs are correctly shaped
        if query.dim() == 2:
            query = query.unsqueeze(0)  # [1, batch_size, embed_dim]
        else:
            query = query.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
            
        if key.dim() == 2:
            key = key.unsqueeze(0)  # [1, batch_size, embed_dim]
        elif key.dim() == 3:
            key = key.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        
        if value.dim() == 2:
            value = value.unsqueeze(0)  # [1, batch_size, embed_dim]
        elif value.dim() == 3:
            value = value.transpose(0, 1)  # [seq_len, batch_size, embed_dim]


        query = query.to(self.device)
        key = key.to(self.device)
        value = value.to(self.device)
        
        # Apply multihead attention
        attn_output, _ = self.attention(query, key, value)
        
        # Return to original shape: [batch_size, embed_dim]
        return attn_output.transpose(0, 1).squeeze(0)

class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim=256, device='cuda'):
        super().__init__()
        self.device = device
        self.mha = MultiHeadAttention(d_model=hidden_dim).to(self.device)
        
    def forward(self, text_features, audio_features, video_features):
        # Create sequence for key and value (3 Ã— [batch_size, feature_dim])
        # -> [3, batch_size, feature_dim]
        features = torch.stack([text_features.to(self.device), audio_features.to(self.device), video_features.to(self.device)], dim=0)        
        # Apply attention to each feature vector as query
        # We need to ensure features tensor is correctly shaped for the attention mechanism
        text_attn = self.mha(text_features, features, features)
        audio_attn = self.mha(audio_features, features, features)
        video_attn = self.mha(video_features, features, features)
        
        # Combine attended features
        fused_features = (text_attn + audio_attn + video_attn) / 3
        
        return fused_features

# %%
class SimplerAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=256):
        super(SimplerAttentionFusion, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 3)
        
    def forward(self, text_features, audio_features, video_features):
        # Stack features along a new dimension
        features = torch.stack([text_features, audio_features, video_features], dim=1)  # [batch_size, 3, hidden_dim]
        
        # Calculate attention weights (simplified attention)
        batch_size = features.size(0)
        
        # Use the mean of all features as a query
        query = features.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Calculate attention scores
        attention_scores = self.attention_weights(query)  # [batch_size, 3]
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(2)  # [batch_size, 3, 1]
        
        # Apply attention weights
        weighted_features = features * attention_weights  # [batch_size, 3, hidden_dim]
        
        # Sum over the modalities
        fused_features = weighted_features.sum(dim=1)  # [batch_size, hidden_dim]
        
        return fused_features

# ==================== FULL MODEL ====================
class MultimodalEmotionRecognition(nn.Module):
    def __init__(self, hidden_dim=256):
        super(MultimodalEmotionRecognition, self).__init__()
        
        # Encoders for each modality
        self.text_encoder = TextEncoder(hidden_size=768, pretrained_model="distilbert-base-uncased")
        self.audio_encoder = AudioEncoder(hidden_dim=hidden_dim)
        self.video_encoder = VideoEncoder(hidden_dim=hidden_dim)

         # Use the simpler attention fusion module
        self.fusion = SimplerAttentionFusion(hidden_dim=hidden_dim)
        
        # Final classification layer
        self.fc = nn.Linear(hidden_dim, 1)  # Binary classification for surprise
        
    def forward(self, texts, audio_specs, video_frames):
        # Fix video input dimensions if needed
        if video_frames.dim() == 5:  # [batch_size, 1, 3, height, width]
            video_frames = video_frames.squeeze(1)
            
        # Encode each modality
        text_features = self.text_encoder(texts)
        audio_features = self.audio_encoder(audio_specs)
        video_features = self.video_encoder(video_frames)
        
        # Fuse features using attention
        fused_features = self.fusion(text_features, audio_features, video_features)
        
        # Classification
        output = self.fc(fused_features)
        output = torch.sigmoid(output)  # Probability of surprise emotion
        
        # Ensure output maintains proper dimensions for batch size 1
        batch_size = text_features.size(0)
        output = output.view(batch_size)  # Reshape to [batch_size]
        
        return output



def predict_emotion(model, text, audio_path, video_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Move model to GPU before prediction
    model.eval()
    
    with torch.no_grad():
        try:
            # Preprocess inputs
            audio_spec = AudioEncoder.extract_spectrogram(audio_path).to(device)
            video_features = VideoEncoder.extract_face_features(video_path).to(device)
            
            # Ensure proper batch dimension
            if audio_spec.dim() == 3:  # [channel, height, width]
                audio_spec = audio_spec.unsqueeze(0)  # Add batch dimension -> [batch_size, channel, height, width]
                
            if video_features.dim() == 3:  # [channel, height, width]
                video_features = video_features.unsqueeze(0)  # Add batch dimension
            
            # Forward pass
            output = model([text], audio_spec, video_features)
            
            # Get prediction
            if output.dim() == 0:  # If scalar
                probability = output.item()
            else:
                probability = output.squeeze().item()
                
            prediction = "Surprise" if probability > 0.5 else "Not Surprise"
            
            return prediction, probability
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            print(f"Audio spec shape: {audio_spec.shape}")
            print(f"Video features shape: {video_features.shape}")
            raise e

# %%
import torch

# 1. Load the saved model
def load_trained_model(model_path, device):
    # Initialize model architecture
    model = MultimodalEmotionRecognition()
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    
    # Set to evaluation mode
    model.eval()
    model = model.to(device)
    return model

# 2. Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Load your pretrained model



import os
import re
import json
import cv2
from datetime import timedelta
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from google.cloud import speech
import pysrt

# ================================================
# === AUDIO AND TRANSCRIPTION ====================
# ================================================

def extract_audio_from_mkv(input_file, mp3_output_path):
    video_clip = VideoFileClip(input_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(mp3_output_path, codec='mp3', bitrate='320k')
    audio_clip.close()
    video_clip.close()

def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_file_path, format="wav")

def format_timedelta(td):
    total_seconds = int(td.total_seconds())
    milliseconds = int(td.microseconds / 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def transcribe_audio_to_srt(wav_file_path, output_srt_path):
    client = speech.SpeechClient()
    with open(wav_file_path, "rb") as audio_file:
        content = audio_file.read()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True
    )
    audio = speech.RecognitionAudio(content=content)
    response = client.recognize(config=config, audio=audio)

    srt_content = []
    index = 1
    for result in response.results:
        for alternative in result.alternatives:
            start_time_seconds = max(0, result.result_end_time.seconds - len(alternative.transcript.split()) * 0.5)
            start_time = timedelta(seconds=start_time_seconds)
            end_time = timedelta(seconds=result.result_end_time.seconds)
            srt_content.append(f"{index}")
            srt_content.append(f"{format_timedelta(start_time)} --> {format_timedelta(end_time)}")
            srt_content.append(alternative.transcript)
            srt_content.append("")
            index += 1

    with open(output_srt_path, "w", encoding="utf-8") as srt_file:
        srt_file.write("\n".join(srt_content))


# ================================================
# === SPLIT VIDEO/AUDIO/TEXT =====================
# ================================================

def split_by_srt(video_path, subtitle_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    subs = pysrt.open(subtitle_path)
    video = VideoFileClip(video_path)

    for sub in subs:
        index = sub.index
        start_time = sub.start.to_time()
        end_time = sub.end.to_time()
        start_sec = start_time.hour * 3600 + start_time.minute * 60 + start_time.second + start_time.microsecond / 1e6
        end_sec = end_time.hour * 3600 + end_time.minute * 60 + end_time.second + end_time.microsecond / 1e6

        clip = video.subclip(start_sec, end_sec)
        clip.write_videofile(os.path.join(output_dir, f"{index}.mp4"), codec='libx264', audio_codec='aac', verbose=False, logger=None)
        clip.audio.write_audiofile(os.path.join(output_dir, f"{index}.wav"), verbose=False, logger=None)
        with open(os.path.join(output_dir, f"{index}.txt"), 'w', encoding='utf-8') as f:
            f.write(sub.text)

    video.close()


# ================================================
# === INFERENCE ===================================
# ================================================

def extract_mid_frame(video_path, output_image_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame_num = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_num)
    success, frame = cap.read()
    if success:
        cv2.imwrite(output_image_path, frame)
    cap.release()

def parse_srt(srt_path):
    with open(srt_path, "r", encoding="utf-8") as f:
        srt_data = f.read()
    entries = []
    blocks = re.split(r'\n\s*\n', srt_data.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) >= 3:
            id_ = lines[0].strip()
            times = lines[1].strip()
            caption = " ".join(line.strip() for line in lines[2:])
            start_time, end_time = times.split(" --> ")
            entries.append({
                "id": id_,
                "start_time": start_time,
                "end_time": end_time,
                "caption": caption
            })
    return entries

def batch_infer_srt_format(model, folder_path, srt_path, output_json_path, image_output_dir):
    os.makedirs(image_output_dir, exist_ok=True)
    srt_entries = parse_srt(srt_path)
    final_results = []

    for entry in srt_entries:
        id_ = entry["id"]
        txt_path = os.path.join(folder_path, f"{id_}.txt")
        wav_path = os.path.join(folder_path, f"{id_}.wav")
        mp4_path = os.path.join(folder_path, f"{id_}.mp4")
        image_path = os.path.join(image_output_dir, f"{id_}_mid.jpg")

        try:
            caption = entry["caption"]
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()

            prediction, probability = predict_emotion(model, caption, wav_path, mp4_path)
            extract_mid_frame(mp4_path, image_path)

            final_results.append({
                "id": id_,
                "start_time": entry["start_time"],
                "end_time": entry["end_time"],
                "caption": caption,
                "surprise": "1" if prediction.lower() == "surprise" else "0",
                "probability": round(probability, 4),
                "midimage": image_path.replace("\\", "/")
            })

        except Exception as e:
            print(f"Error processing ID {id_}: {e}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"JSON saved to: {output_json_path}")


# ================================================
# === MAIN DRIVER ================================
# ================================================

def full_pipeline(video_path, model, output_json_path="inference_results.json"):
    base = os.path.splitext(os.path.basename(video_path))[0]
    mp3_path = f"{base}.mp3"
    wav_path = f"{base}_temp.wav"
    srt_path = f"{base}.srt"
    output_dir = "output"
    image_output_dir = "midimages"

    if not os.path.exists(srt_path):
        print("[1/4] Extracting audio from video...")
        extract_audio_from_mkv(video_path, mp3_path)

        print("[2/4] Converting audio to WAV...")
        convert_mp3_to_wav(mp3_path, wav_path)

        print("[3/4] Generating SRT via transcription...")
        transcribe_audio_to_srt(wav_path, srt_path)
        os.remove(wav_path)
        
    
    print("[4/4] Splitting video/audio/text chunks...")
    split_by_srt(video_path, srt_path, output_dir)

    print("[5/5] Running inference...")
    batch_infer_srt_format(model, output_dir, srt_path, output_json_path, image_output_dir)

    print("✅ Pipeline completed.")



import tempfile


st.set_page_config(page_title="Multimodal Emotion Inference", layout="wide")

st.title("🎬 Multimodal Surprise Recognition Pipeline")
st.write("Upload a `.mkv` video file to analyze emotions using text, audio, and visual cues.")

# === Upload Section
uploaded_file = st.file_uploader("Upload an MKV video", type=["mkv"])



## change the model location according to where you run it from


#################################################################################################################
model = load_trained_model('/Users/akhsinak/co/nlp_project/models/best_model_hubert.pth', device)
#################################################################################################################

if uploaded_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        st.info("Saving uploaded video...")
        video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        json_output_path = os.path.join(temp_dir, "inference_results.json")

        with st.spinner("Running full pipeline... ⏳"):
            full_pipeline(video_path, model, json_output_path)

        st.success("✅ Inference completed!")

        # === Load results
        with open(json_output_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        st.subheader("📄 Inference Results")
        for entry in results:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(entry["midimage"], caption=f"Clip {entry['id']}", width=200)
            with col2:
                st.markdown(f"""
                **Caption**: {entry['caption']}  
                **Start**: {entry['start_time']} | **End**: {entry['end_time']}  
                **Emotion**: `{entry['surprise']}`  
                **Probability**: `{entry['probability']}`
                """)

        st.download_button(
            "📥 Download JSON Results",
            data=json.dumps(results, indent=2, ensure_ascii=False),
            file_name="inference_results.json",
            mime="application/json"
        )

