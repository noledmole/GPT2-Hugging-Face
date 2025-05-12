import argparse
import time
import csv
import psutil
import subprocess
import os
import datetime

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

def load_sample_dataset(sample_size: int):
    ds = load_dataset("noelmurti/spotify_data", split="train")
    ds = ds.filter(lambda row: row.get("song") and row.get("Artist(s)"))
    ds = ds.map(lambda row: {"text": f"{row['song']} - {row['Artist(s)']}"})
    return ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))

def train_model(sample_size: int):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    usage_log = f"logs/usage_monitor_{sample_size}_{timestamp}.csv"

    monitor_process = subprocess.Popen(["python", "monitor.py", "--logfile", usage_log])
    
    dataset = load_sample_dataset(sample_size)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir="./models/gpt2-spotify",
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        report_to="none",
        run_name="gpt2-spotify-run"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    cpu_start = psutil.cpu_percent()
    mem_start = psutil.virtual_memory().percent
    start_time = time.time()

    trainer.train() 

    end_time = time.time()
    cpu_end = psutil.cpu_percent()
    mem_end = psutil.virtual_memory().percent
    training_time = round(end_time - start_time, 2)

    time.sleep(2)
    monitor_process.terminate()
    monitor_process.wait()


    model.save_pretrained("./models/gpt2-spotify")
    tokenizer.save_pretrained("./models/gpt2-spotify")

    os.makedirs("logs", exist_ok=True)
    training_log_file = "logs/training_times.csv"
    if not os.path.exists(training_log_file):
        with open(training_log_file, "w") as f:
            f.write("sample_size,timestamp,training_time_sec,cpu_start_percent,cpu_end_percent,mem_start_percent,mem_end_percent\n")

    with open(training_log_file, "a") as f:
        f.write(f"{sample_size},{timestamp},{training_time},{cpu_start},{cpu_end},{mem_start},{mem_end}\n")

    print(f"Training complete. Time: {training_time}s | Log saved to {training_log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10000, help="Sample size from dataset")
    args = parser.parse_args()
    train_model(args.size)

