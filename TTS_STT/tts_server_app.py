from fastapi import FastAPI, UploadFile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import torchaudio
import sys
import uvicorn

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Whisper model
model_id = "openai/whisper-base"

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


@app.post("/transcribe")
async def transcribe(file: UploadFile):
    audio_path = f"temp_{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    waveform, sample_rate = torchaudio.load(audio_path)
    result = pipe({"array": waveform.squeeze().numpy(), "sampling_rate": sample_rate}, return_timestamps=True)
    return {"text": result["text"]}


if __name__ == "__main__":
    # Disable autoreload in the compiled executable
    if getattr(sys, "frozen", False):
        # 🧊 Running from EXE — run directly with app object
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    else:
        # 🧩 Running from Python — use module import so autoreload works
        uvicorn.run("tts_server_app:app", host="0.0.0.0", port=8000, reload=True)
