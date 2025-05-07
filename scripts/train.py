import os
import time
import argparse
import psutil
import pandas as pd
import kagglehub
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling)


def load_sample_dataset(sample_size: int) -> Dataset:
    print("Downloading dataset from Kaggle...")
    data_dir = kagglehub.dataset_download("devdope/900k-spotify")
    print(f"Dataset downloaded to: {data_dir}")

    csv_path = os.path.join(data_dir, "tracks.csv")
    print(f"Reading CSV file from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total rows in dataset: {len(df)}")

    df_sample = df.sample(n=sample_size, random_state=42)
    df_sample['text'] = df_sample['track_name'] + ' - ' + df_sample['artist_name']
    print(f"Sampled {sample_size} rows")

    return Dataset.from_pandas(df_sample[['text']])


def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)


def monitor_resources():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    return cpu, mem


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
    cpu_start, mem_start = monitor_resources()

    trainer.train()

    end_time = time.time()
    cpu_end, mem_end = monitor_resources()

    training_time = round(end_time - start_time, 2)
    print(f"Training time for {sample_size} samples: {training_time}s")

    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', 'training_times.csv')
    with open(log_path, 'a') as f:
        f.write(f"{sample_size},{training_time},{cpu_start},{cpu_end},{mem_start},{mem_end}\n")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=10000, help='Sample size of Spotify dataset')
    args = parser.parse_args()

    train_model(args.size)
