import sys
import numpy as np
import joblib
from pathlib import Path


MODEL_PATH = Path("resources/model.pkl")


def main():
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python app.py path/to/x_data.npy")

    x_path = Path(sys.argv[1])

    X = np.load(x_path)
    model = joblib.load(MODEL_PATH)

    preds = model.predict(X)
    preds = preds.astype(float)

    print(preds.tolist())


if __name__ == "__main__":
    main()
