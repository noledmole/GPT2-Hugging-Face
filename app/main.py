from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = FastAPI()

model_dir = os.getenv("MODEL_DIR", "models/gpt2-spotify-10000")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class Prompt(BaseModel):
    text: str

@app.post("/generate")
def generate(prompt: Prompt):
    try:
        inputs = tokenizer(prompt.text, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.8)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
