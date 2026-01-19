import os
import sys


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
