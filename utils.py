import os
import sys

import numpy as np


def validate_arguments():
    dataset_path = None
    model_path = None

    if "--dataset" in sys.argv:
        dataset_idx = sys.argv.index("--dataset")
        if dataset_idx + 1 < len(sys.argv):
            dataset_path = sys.argv[dataset_idx + 1]
            if not os.path.exists(dataset_path):
                sys.exit(1)
            if not os.path.isdir(dataset_path):
                sys.exit(1)
            sys.argv.pop(dataset_idx)
            sys.argv.pop(dataset_idx)
        else:
            sys.exit(1)

    if "--model" in sys.argv:
        model_idx = sys.argv.index("--model")
        if model_idx + 1 < len(sys.argv):
            model_path = sys.argv[model_idx + 1]
            model_dir = os.path.dirname(model_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
            sys.argv.pop(model_idx)
            sys.argv.pop(model_idx)
        else:
            sys.exit(1)

    if "--translate" in sys.argv:
        translate_idx = sys.argv.index("--translate")
        if translate_idx + 1 < len(sys.argv):
            translate_text = sys.argv[translate_idx + 1]
            sys.argv.pop(translate_idx)
            sys.argv.pop(translate_idx)
        else:
            sys.exit(1)
    else:
        translate_text = None

    return dataset_path, model_path, translate_text


def split_train_val(encoded_en, encoded_vi, split_ratio=0.8):
    total_samples = len(encoded_en)
    split_idx = int(split_ratio * total_samples)

    indices = np.random.permutation(total_samples)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    X_train = [encoded_en[i] for i in train_indices]
    Y_train = [encoded_vi[i] for i in train_indices]
    X_val = [encoded_en[i] for i in val_indices]
    Y_val = [encoded_vi[i] for i in val_indices]

    return X_train, Y_train, X_val, Y_val
