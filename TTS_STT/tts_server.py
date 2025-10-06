from fastapi import FastAPI, UploadFile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import torchaudio

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# https://huggingface.co/openai/whisper-tiny
model_id = "openai/whisper-base" # tiny, base 

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# run with uvicorn tts_server:app --host 0.0.0.0 --port 8000 --reload

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    audio_path = f"temp_{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # Load with torchaudio (already works fine)
    waveform, sample_rate = torchaudio.load(audio_path)

    # Pass raw waveform to the pipeline instead of file path
    result = pipe({"array": waveform.squeeze().numpy(), "sampling_rate": sample_rate}, return_timestamps=True)
    return {"text": result["text"]}
