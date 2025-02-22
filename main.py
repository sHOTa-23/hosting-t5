from fastapi import FastAPI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from pydantic import BaseModel
from typing import List
import time


app = FastAPI()

MODEL_NAME = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = model.to(device)


class PromptRequest(BaseModel):
    prompt: str

def generate_text(prompt: str, max_length: int = 50):
    print("DASDASDASD")
    start_time = time.time()
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    end_time = time.time()
    print(f"Computation Time: {end_time - start_time:.4f} seconds")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.post("/generate")
def generate(request: PromptRequest):
    response = generate_text(request.prompt)
    return {"response": response}


def generate_text_batch(prompts: List[str], max_length: int = 50):
    start_time = time.time()
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    end_time = time.time()
    print(f"Computation Time: {end_time - start_time:.4f} seconds")
    return responses


class BatchRequest(BaseModel):
    prompts: List[str]

@app.post("/generateBatch")
def generate(request: BatchRequest):
    responses = generate_text_batch(request.prompts)
    return {"responses": responses}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
