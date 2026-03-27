import torch
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor


device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen2-Audio-7B-Instruct"


processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

def analyze_audio_quality(audio_path: str):
 
    audio, sr = librosa.load(audio_path, sr=16000)
    
   
    prompt = (
        "<|audio_pad|>Analyze the perceptual quality of this speech audio. "
        "Please provide: "
        "1. Background Noise Level (1-5) "
        "2. Voice Clarity (1-5) "
        "3. Presence of Artifacts (e.g., robotic sound, clipping) "
        "4. Final Estimated MOS Score (1.0 to 5.0) "
        "5. Brief Reasoning."
    )

   
    messages = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": prompt}
        ]}
    ]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    audios = [audio] 
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

  
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False # Greedy search for consistent ratings
        )
        
        # Trim the input tokens from the output
        generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
        
        response = processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

    return response


if __name__ == "__main__":
    example_audio = "/Users/avataar/Downloads/messy_mashup/mashups/song1487.wav"
   
    result = analyze_audio_quality(example_audio)
    print("\nModel Output:\n", result)
