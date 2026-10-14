from pathlib import Path
import time
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "task1_dataset" / "processed"
OUT_DIR = Path(__file__).resolve().parent / "models"
CHECKPOINT = "distilbert-base-uncased"

def train_target(target):
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    valid_df = pd.read_csv(DATA_DIR / "validation.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    labels = sorted(train_df[target].dropna().unique().tolist())
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    def encode(frame):
        frame = frame[["text", target]].copy()
        frame[target] = frame[target].map(label_to_id)
        ds = Dataset.from_pandas(frame.rename(columns={target: "labels"}), preserve_index=False)
        return ds.map(lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128), batched=True)

    train_ds = encode(train_df)
    valid_ds = encode(valid_df)
    test_ds = encode(test_df)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=len(labels), id2label=id_to_label, label2id=label_to_id
    )
    def metrics(eval_pred):
        logits, gold = eval_pred
        pred = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(gold, pred),
            "macro_f1": f1_score(gold, pred, average="macro", zero_division=0)
        }

    args = TrainingArguments(
        output_dir=str(OUT_DIR / f"work_{target}"),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to="none"
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds, compute_metrics=metrics)
    start = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - start
    valid_metrics = trainer.evaluate(valid_ds)
    test_metrics = trainer.evaluate(test_ds)
    OUT_DIR.mkdir(exist_ok=True)
    save_path = OUT_DIR / target
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    with (OUT_DIR / f"{target}_results.txt").open("w", encoding="utf-8") as f:
        f.write(f"Target: {target}\nTraining seconds: {training_seconds:.2f}\n")
        f.write(f"Validation metrics: {valid_metrics}\nTest metrics: {test_metrics}\n")
    print(target, "training seconds:", round(training_seconds, 2))
    print("Validation:", valid_metrics)
    print("Test:", test_metrics)

def main():
    if not (DATA_DIR / "train.csv").exists():
        raise FileNotFoundError("Part 1 split CSV files are needed in task1_dataset/processed/.")
    for target in ["category", "priority"]:
        train_target(target)

if __name__ == "__main__":
    main()
