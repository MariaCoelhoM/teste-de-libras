import argparse
import re

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from libras.training_utils import extract_signer_ids


def load_features(data_path, use_face=True):
    data = np.load(data_path, allow_pickle=True)
    y_raw = data["y"]
    paths = data["paths"] if "paths" in data else None

    if "X_hands" in data:
        X_hands_raw = data["X_hands"]
        N, T = X_hands_raw.shape[0], X_hands_raw.shape[1]
        X_hands = X_hands_raw.reshape(N, T, -1)

        if use_face and "X_face" in data:
            X_face_raw = data["X_face"]
            X_face = X_face_raw.reshape(N, T, -1)
            X = np.concatenate([X_hands, X_face], axis=-1)
        else:
            X = X_hands
    elif "X" in data:
        X_raw = data["X"]
        N, T = X_raw.shape[0], X_raw.shape[1]
        X = X_raw.reshape(N, T, -1)
    else:
        raise KeyError(f"{data_path} nao tem 'X_hands' nem 'X' - formato de .npz nao reconhecido.")

    return X, y_raw, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Mesmo .npz usado no treino")
    parser.add_argument("--model", required=True, help="Arquivo .keras salvo pelo treino")
    parser.add_argument("--test_size", type=float, default=0.3, help="Mesmo valor usado no treino")
    parser.add_argument("--no_face", action="store_true", help="Use se treinou com --no_face")
    parser.add_argument("--signer_regex", default=r"Sinalizador(\d+)", help="Mesmo regex usado no treino")
    args = parser.parse_args()

    X, y_raw, paths = load_features(args.data, use_face=not args.no_face)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    if paths is not None:
        # Mesmo split por sinalizador do train_lstm_hands_face.py / train_lstm.py
        signer_regex = re.compile(args.signer_regex)
        signer_ids = extract_signer_ids(paths, signer_regex)
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
        _, test_idx = next(splitter.split(X, y, groups=signer_ids))
    else:
        print("Aviso: o .npz nao tem 'paths' - nao da pra reproduzir o split por sinalizador. "
              "Usando split aleatorio por amostra como aproximacao (pode nao bater exatamente "
              "com o conjunto de teste do treino original).")
        from sklearn.model_selection import train_test_split
        _, test_idx = train_test_split(
            np.arange(len(y)), test_size=args.test_size, stratify=y, random_state=42
        )

    X_test, y_test = X[test_idx], y[test_idx]

    model = tf.keras.models.load_model(args.model)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    classes = encoder.classes_
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusao")
    plt.tight_layout()
    plt.savefig("matriz_confusao.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
