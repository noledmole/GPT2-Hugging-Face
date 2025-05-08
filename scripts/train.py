import argparse
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

    trainer.train()
    model.save_pretrained("./models/gpt2-spotify")
    tokenizer.save_pretrained("./models/gpt2-spotify")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10000, help="Sample size from dataset")
    args = parser.parse_args()
    train_model(args.size)

