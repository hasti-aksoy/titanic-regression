# src/predict.py
import argparse, joblib, pandas as pd
from pathlib import Path

def main(model_path, input_csv, output_csv):
    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= 0.5).astype(int)
    df_out = df.copy()
    df_out["survival_proba"] = probs
    df_out["survived_pred"] = preds
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/best_pipeline.joblib")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="../data/processed/predictions.csv")
    args = ap.parse_args()
    main(args.model, args.input, args.output)
