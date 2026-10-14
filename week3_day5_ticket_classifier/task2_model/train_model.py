
from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "task1_dataset" / "processed"
OUT = Path(__file__).resolve().parent
MODEL_DIR = OUT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def build_pipeline(analyzer="word"):
    if analyzer == "char":
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
    else:
        vectorizer = TfidfVectorizer()
    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

def train_one(target, train_df, valid_df):
    pipe = build_pipeline()
    params = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier__C": [0.5, 1.0, 2.0]
    }
    search = GridSearchCV(pipe, params, cv=3, scoring="f1_macro", n_jobs=-1)
    search.fit(train_df["text"], train_df[target])
    model = search.best_estimator_
    actual = valid_df[target]
    predicted = model.predict(valid_df["text"])
    accuracy = accuracy_score(actual, predicted)
    macro_f1 = f1_score(actual, predicted, average="macro", zero_division=0)
    labels = sorted(actual.unique())
    matrix = confusion_matrix(actual, predicted, labels=labels)
    report = [
        f"Target: {target}",
        f"Best parameters: {search.best_params_}",
        f"Validation accuracy: {accuracy:.3f}",
        f"Validation macro F1: {macro_f1:.3f}",
        "Labels: " + ", ".join(labels),
        "Confusion matrix (rows=actual, columns=predicted):",
        str(matrix),
        "",
        "Classification report:",
        classification_report(actual, predicted, zero_division=0)
    ]
    (OUT / f"{target}_metrics.txt").write_text("\n".join(report), encoding="utf-8")
    joblib.dump(model, MODEL_DIR / f"{target}_model.joblib")

    # Save ten lowest-confidence validation predictions as simple error-analysis examples.
    probabilities = model.predict_proba(valid_df["text"])
    confidence = probabilities.max(axis=1)
    errors = pd.DataFrame({
        "text": valid_df["text"].values,
        "actual": actual.values,
        "predicted": predicted,
        "confidence": confidence
    })
    errors["guess_why"] = "The ticket wording may be short, generic, or share words with another class."
    errors = errors.sort_values("confidence").head(10)
    errors.to_csv(OUT / f"{target}_confusing_examples.csv", index=False)
    print("\n".join(report[:5]))
    return model, accuracy, macro_f1

def main():
    if not (DATA_DIR / "train.csv").exists():
        from task1_dataset.prepare_data import main as prepare
        prepare()
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    valid_df = pd.read_csv(DATA_DIR / "validation.csv")
    print("Training category model...")
    category_model, cat_acc, cat_f1 = train_one("category", train_df, valid_df)
    print("\nTraining priority model...")
    priority_model, pri_acc, pri_f1 = train_one("priority", train_df, valid_df)

   
    bonus_lines = []
    for target, word_model in [("category", category_model), ("priority", priority_model)]:
        char_model = build_pipeline("char")
        char_model.fit(train_df["text"], train_df[target])
        pred = char_model.predict(valid_df["text"])
        char_acc = accuracy_score(valid_df[target], pred)
        char_f1 = f1_score(valid_df[target], pred, average="macro", zero_division=0)
        bonus_lines.append(
            f"{target}: word accuracy={accuracy_score(valid_df[target], word_model.predict(valid_df['text'])):.3f}, "
            f"word macro F1={f1_score(valid_df[target], word_model.predict(valid_df['text']), average='macro', zero_division=0):.3f}; "
            f"char accuracy={char_acc:.3f}, char macro F1={char_f1:.3f}"
        )
    (OUT / "bonus_comparison.txt").write_text("\n".join(bonus_lines), encoding="utf-8")
    print("\nSaved models and reports. Test split is kept separate for later final testing.")

if __name__ == "__main__":
    main()
