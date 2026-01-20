import os
import sys

import numpy as np


def validate_arguments():
    dataset_path = None
    model_path = None
    architecture = "lstm-lstm"
    device = "cpu"

    if "--device" in sys.argv:
        dev_idx = sys.argv.index("--device")
        if dev_idx + 1 < len(sys.argv):
            requested_device = sys.argv[dev_idx + 1].lower()
            if requested_device not in ["cpu", "gpu"]:
                print("Invalid device. Use 'cpu' or 'gpu'. Defaulting to 'cpu'.")
                device = "cpu"
            elif requested_device == "gpu":
                # Check if GPU is actually available
                try:
                    import cupy as cp
                    # Try to create array and perform operation to verify CUDA works
                    test = cp.array([1.0, 2.0, 3.0])
                    result = cp.sum(test)
                    # Force computation
                    _ = float(result)
                    device = "gpu"
                    print("[OK] GPU detected and available")
                except ImportError:
                    print("[WARNING] GPU requested but CuPy not installed")
                    print("  Falling back to CPU mode")
                    print("  To use GPU: pip install cupy-cuda12x (requires CUDA Toolkit)")
                    device = "cpu"
                except Exception as e:
                    error_msg = str(e)
                    if "nvrtc64" in error_msg or "CUDA" in error_msg:
                        print("[WARNING] GPU requested but CUDA Toolkit not installed")
                        print("  Falling back to CPU mode")
                        print("  To use GPU: Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
                    else:
                        print(f"[WARNING] GPU error: {error_msg[:80]}")
                        print("  Falling back to CPU mode")
                    device = "cpu"
            else:
                device = requested_device

    if "--architecture" in sys.argv:
        arch_idx = sys.argv.index("--architecture")
        if arch_idx + 1 < len(sys.argv):
            arch_value = sys.argv[arch_idx + 1].lower()
            if arch_value in ["lstm-lstm", "seq2seq"]:
                architecture = arch_value
            else:
                print("Error: --architecture must be 'lstm-lstm' or 'seq2seq'")
                sys.exit(1)
            sys.argv.pop(arch_idx)
            sys.argv.pop(arch_idx)
        else:
            print("Error: --architecture requires a value")
            sys.exit(1)

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

    if not dataset_path and not translate_text:
        print("Error: Must provide --dataset for training or --translate for translation.")
        sys.exit(1)

    return dataset_path, model_path, translate_text, architecture, device


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
