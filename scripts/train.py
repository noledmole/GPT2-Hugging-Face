import os
import time
import argparse
import psutil
from datasets import Dataset, load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling)
import subprocess


def load_sample_dataset(sample_size: int) -> Dataset:
    print("Loading dataset from Hugging Face Hub...")
    ds = load_dataset("noelmurti/spotify_data", split="train")
    print(f"Loaded {len(ds)} rows from HF")

    ds = ds.shuffle(seed=42).select(range(sample_size))
    ds = ds.map(lambda row: {"text": row["track_name"] + " - " + row["artist_name"]})
    return ds


def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)


def monitor_resources():
    import subprocess
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    try:
        nvidia_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,nounits,noheader']
        ).decode().strip()
        gpu_util, gpu_mem = map(int, nvidia_output.split(','))
    except:
        gpu_util, gpu_mem = -1, -1
    return cpu, mem, gpu_util, gpu_mem


def train_model(sample_size: int):
    output_dir = os.path.join('models', f'gpt2-spotify-{sample_size}')

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2')

    dataset = load_sample_dataset(sample_size)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=1,
        logging_steps=50,
        evaluation_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    start_time = time.time()
    cpu_start, mem_start, gpu_start, gpumem_start = monitor_resources()

    trainer.train()

    end_time = time.time()
    cpu_end, mem_end, gpu_end, gpumem_end = monitor_resources()

    training_time = round(end_time - start_time, 2)
    print(f"Training time for {sample_size} samples: {training_time}s")

    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', 'training_times.csv')
    with open(log_path, 'a') as f:
        f.write(f"{sample_size},{training_time},{cpu_start},{cpu_end},{mem_start},{mem_end},{gpu_start},{gpu_end},{gpumem_start},{gpumem_end}\n")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=10000, help='Sample size of Spotify dataset')
    args = parser.parse_args()

    train_model(args.size)
