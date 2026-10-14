from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
OUT = HERE / "processed"

def main():
    OUT.mkdir(exist_ok=True)
    source = HERE / "tickets.csv"
    if not source.exists():
        from generate_dataset import main as generate
        generate()
    data = pd.read_csv(source)
    data["text"] = data["text"].fillna("").str.lower().str.strip()
    data["category"] = data["category"].fillna("").str.strip()
    data["priority"] = data["priority"].fillna("").str.strip()
    data = data.drop_duplicates(subset=["text", "category", "priority"]).reset_index(drop=True)

    print(f"Rows after cleaning: {len(data)}")
    print("\nCategory counts:")
    print(data["category"].value_counts().to_string())
    print("\nPriority counts:")
    print(data["priority"].value_counts().to_string())

    # Stratify by category and priority together to keep both label distributions similar.
    data["strata"] = data["category"].astype(str) + "_" + data["priority"].astype(str)
    train, temp = train_test_split(data, test_size=0.30, random_state=42, stratify=data["strata"])
    valid, test = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp["strata"])
    for part in (train, valid, test):
        part.drop(columns=["strata"]).to_csv(OUT / ("train.csv" if part is train else "validation.csv" if part is valid else "test.csv"), index=False)
    data.to_csv(OUT / "cleaned_tickets.csv", index=False)
    print(f"\nSplit sizes: train={len(train)}, validation={len(valid)}, test={len(test)}")
    print(f"Saved files in: {OUT}")

if __name__ == "__main__":
    main()
