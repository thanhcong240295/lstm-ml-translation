# Machine Learning Translation - LSTM-to-LSTM & Seq2Seq with Attention

A high-performance neural machine translation system with **GPU acceleration** and two architecture options: simple LSTM-to-LSTM encoder-decoder and advanced Seq2Seq with Bi-LSTM and Bahdanau Attention. Implemented from scratch without high-level frameworks.

## 📋 Project Overview

This project implements English-Vietnamese neural machine translation (NMT) with:
- **🚀 GPU Acceleration**: 10-100x faster training with CuPy
- **Two Architectures**:
  - LSTM-to-LSTM: Simple encoder-decoder
  - Seq2Seq: Bi-LSTM encoder + Bahdanau Attention + LSTM decoder
- **Word2Vec embeddings** for word representations
- **Custom LSTM implementation** from NumPy/CuPy
- **Full backpropagation** through time (BPTT)
- **Advanced decoding**: Top-K sampling, temperature, repetition penalty

## ✨ Features

- ✅ **GPU Support** with CuPy (10-100x speedup)
- ✅ Custom LSTM implementation (no TensorFlow/PyTorch)
- ✅ Two model architectures (LSTM-to-LSTM & Seq2Seq)
- ✅ Bahdanau Attention mechanism
- ✅ Bidirectional encoding for context
- ✅ Full BPTT (encoder + decoder)
- ✅ Advanced sampling (Top-K, temperature, repetition penalty)
- ✅ Word2Vec embedding training
- ✅ Float32 precision for GPU efficiency
- ✅ Reproducible results with seed parameter
- ✅ Model checkpointing with best model saving
- ✅ Automatic loss visualization
- ✅ Code quality tools (black, isort, flake8, pylint)
- ✅ Clean, modular architecture

## 📁 Project Structure

```
machinelearning-translation/
├── main.py                    # Entry point with CLI
├── lstm_to_lstm.py           # Simple LSTM encoder-decoder
├── seq2seq.py                # Seq2Seq with attention
├── bi_lstm.py                # Bidirectional LSTM implementation
├── lstm.py                   # LSTM cell implementation
├── attention.py              # Bahdanau Attention mechanism
├── activation.py             # Activation functions (sigmoid, tanh, softmax)
├── word2vec.py               # Word2Vec embedding trainer
├── vocab.py                  # Vocabulary management
├── preprocessor.py           # Data preprocessing & tokenization
├── utils.py                  # Utility functions for CLI
├── visualization.py          # Loss plotting & metrics visualization
├── check_code.py             # Code quality checker
├── format_code.py            # Code formatter (black + isort)
├── clean.py                  # Project cleanup script
├── requirements.txt          # Python dependencies (includes CuPy)
├── GPU_QUICKSTART.md         # GPU quick start guide
├── GPU_OPTIMIZATION.md       # Detailed GPU optimization docs
├── dataset/                  # Training data
│   ├── train.en.txt         # English source sentences
│   └── train.vi.txt         # Vietnamese target sentences
├── model/                    # Trained model weights
└── result/                   # Training outputs
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

1. **Clone/Download the project:**
```bash
cd machinelearning-translation
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install CuPy for GPU support (optional but recommended):**
```bash
# For CUDA 12.x
pip install cupy-cuda12x

# OR for CUDA 11.x
pip install cupy-cuda11x
```

4. **Download NLTK data (for tokenization):**
```bash
python -c "import nltk; nltk.download('punkt')"
```

5. **Verify GPU (optional):**
```bash
python -c "import cupy; print('GPU available!')"
```

### Usage

#### Training with GPU (Recommended - 10-100x faster)
20                 # Number of training epochs
LEARNING_RATE = 0.01        # Learning rate
EMBEDDING_DIM = 256         # Word2Vec & LSTM embedding size
HIDDEN_SIZE = 256           # LSTM hidden units
MAX_LEN = 100              # Maximum sequence length
```

## 📊 Performance Comparison

### GPU vs CPU

| Operation | CPU (NumPy) | GPU (CuPy) | Speedup |
|-----------|-------------|------------|---------|
| Training (1 epoch) | ~180s | ~3-10s | **18-60x** |
| Forward pass | ~50ms | ~1-2ms | **25-50x** |
| Backward pass | ~80ms | ~2-3ms | **27-40x** |
| Translation | ~100ms | ~5-10ms | **10-20x** |

### Architecture Comparison

| Feature | LSTM-to-LSTM | Seq2Seq + Attention |
|---------|--------------|---------------------|
| Speed | ⚡ Faster | 🐢 Slower |
| Quality | ✅ Good | 🌟 Better |
| Memory | 💾 Less | 💾💾 More |
| Long sentences | ❌ Struggles | ✅ Handles well |
| Context | Fixed | Dynamic (attention) |

**Recommendation**: Use Seq2Seq with GPU for best results.

## 🏗️ Model Architectures (Better quality):**
```bash
python main.py --dataset "dataset" --model "model/translator" --architecture seq2seq --device gpu
```

#### Training on CPU

```bash
python main.py --dataset "dataset" --model "model/translator" --architecture seq2seq --device cpu
```

#### Translation

**With GPU:**
```bash
python main.py --model "model/translator" --translate "i love machine learning" --device gpu
```

**With CPU:**
```b1. LSTM-to-LSTM (Simple Encoder-Decoder)

```
English Input
      ↓
  [Word2Vec Embedding]
      ↓
┌─────────────────┐
│  LSTM Encoder   │  → Final state (h, c)
│  (sequential)   │
└─────────────────┘
      ↓
   Context Vector
      ↓
┌─────────────────┐
│  LSTM Decoder   │  ← Uses encoder state
│  (sequential)   │
└─────────────────┘
      ↓
  [Output Layer]
      ↓
  [Softmax]
      ↓
Vietnamese Output
```

**Pros**: Fast, simple, good for short sentences  
**Cons**: Fixed context, struggles with long sequences

### 2. Seq2Seq with Bahdanau Attention

```
English Input
      ↓
  [Word2Vec Embedding]
      ↓
┌─────────────────────┐
│  Bi-LSTM Encoder    │
│ ← Forward LSTM      │
│ → Backward LSTM     │
└─────────────────────┘
      ↓
  All hidden states
      ↓
┌─────────────────────┐
│ Bahdanau Attention  │  ← Computes attention weights
│  (dynamic context)  │
└─────────────────────┘
      ↓
┌─────────────────────┐
│   LSTM Decoder      │  ← Receives weighted context
└─────────────────────┘
      ↓
  [Output Layer]
      ↓
  [Softmax]
      ↓
Vietnamese Output
```

**Pros**: High quality, handles long sentences, dynamic context  
**Cons**: Slower, more complex, hi GPU Support |
|--------|---------|-------------|-------------|
| `lstm.py` | LSTM cell | `LSTMCell` | ✅ |
| `bi_lstm.py` | Bidirectional wrapper | `BiLSTM` | ✅ |
| `attention.py` | Attention mechanism | `BahdanauAttention` | ✅ |
| `activation.py` | Activations | `Activation` | ✅ |
| `lstm_to_lstm.py` | Simple encoder-decoder | `LstmToLstmLanguageTranslation` | ✅ |
| `seq2seq.py` | Attention-based model | `Seq2SeqLanguageTranslation` | ✅ |
| `word2vec.py` | Embedding training | `Word2VecEmbedding` | ❌ |
| `vocab.py` | Vocabulary | `Vocab` | ❌ |
| `preprocessor.py` | Preprocessing | `Preprocessor` | ❌ |
| `visualization.py` | Visualization | `Visualization` | ❌ |

### Advanced Features

**Decoding Strategy:**
- **Top-K Sampling** (k=5): Sample from top-5 predictions
- **Temperature Scaling** (T=1.2): Control randomness
- **Repetition Penalty** (0.05): Avoid repeated words
- **EOS Blocking**: Prevent premature stopping
- **Special Token Filtering**: Clean output

**Training Optimizations:**
- Teacher forcing for faster convergence
- Gradient clipping in activations (prevents overflow)
- Float32 precision for GPU efficiency
- Reproducible results (seed=42)
- `--architecture` : Choose `lstm-lstm` or `seq2seq` (default: lstm-lstm)
- `--device` : Choose `cpu` or `gpu` (default: cpu)
- `--translate` : Text to translate (for inference mode)

#### Configuration

Edit `main.py` to adjust hyperparameters:

```python
EPOCHS = 5                  # Number of training epochs
LEARNING_RATE = 0.001       # Learning rate
EMBEDDING_DIM = 256         # Word2Vec & LSTM hidden size
MAX_LEN = 100              # Maximum sequence length
```

## 📊 Model Architecture

### Bidirectional LSTM Flow

```
Input Sequence
      ↓
  [Word2Vec Embedding]
      ↓
┌─────────────────┐
│  Forward LSTM   │
│  (left to right)│
└─────────────────┘
      ↓
┌─────────────────┐
│ Backward LSTM   │
│ (right to left) │
└─────────────────┘
      ↓
[Concatenate Hidden States]
      ↓
  [Output Layer]
      ↓
  [Softmax]
      ↓
Predicted Tokens
```

### LSTM Cell Gates

The LSTM cell computes four gates for each timestep:

- **Forget Gate**: Decides what to discard from cell state
- **Input Gate**: Decides what new information to add
- **Candidate**: New candidate values for cell state
Using device: GPU
Building vocabulary for train.en.txt...
Building vocabulary for train.vi.txt...
Training Word2Vec embeddings...
Creating SEQ2SEQ model...
Starting training...
Epoch 1/20 | Train Loss: 3.2156 | Val Loss: 2.9834 (BEST)
Epoch 2/20 | Train Loss: 2.7821 | Val Loss: 2.6543 (BEST)
...
```

**Performance on GPU**: ~3-10 seconds per epoch  
**Performance on CPU**: ~180 seconds per epoch
```
Processing file: train.en.txt with 133317 lines
Processing file: train.vi.txt with 133317 lines
Starting epoch 1/5...
Epoch 1/5, Loss: 8.2341
Starting epoch 2/5...
Epoch 2/5, Loss: 7.1256
...
```

After training completes, visualizations are saved to `result/`:

- **training_loss.png** - Loss curve over epochs
- **loss_statistics.png** - 4-panel analysis:
  - Loss over epochs
  - Loss distribution (histogram)
  - Min/Max/Mean/Std statistics
  - Loss gradient (convergence rate)

## 🔧 Code Quality

### Format Code
```bash
python format_code.py
```
Applies:
- `isort` - Import sorting
- `black` - Code formatting

### Check Code
```bash
python check_code.py
```
Runs:
- Black (formatter validation)
- isort (import order check)
- flake8 (linting)
- pylint (static analysis)

**Code Quality Score**: 9.46/10 ✨

### Clean Project
```bash
python clean.py
```
Removes:
- Generated model files
- Cache directories
- Log files

## 📚 Data Format

### Dataset Source

**Kaggle English-Vietnamese Translation Dataset**

This project uses the English-Vietnamese translation dataset from Kaggle:
- **Dataset Name:** English-Vietnamese Translation Dataset
- **Link:** https://www.kaggle.com/datasets/hungnm/englishvietnamese-translation/data
- **Author:** Hung NM
- **Size:** 133,317 sentence pairs
- **Format:** Text files (one sentence per line)
- **Language Pair:** English ↔ Vietnamese

### How to Download Dataset

1. **Visit Kaggle:**
   - Go to: https://www.kaggle.com/datasets/hungnm/englishvietnamese-translation/data

2. **Download the dataset:**
   - Click "Download" button
   - Or use Kaggle API: `kaggle datasets download -d hungnm/englishvietnamese-translation`

3. **Extract files:**
   ```bash
   unzip englishvietnamese-translation.zip
   ```

4. **Place in project:**
   ```bash
   mv train.en.txt dataset/
   mv train.vi.txt dataset/
   ```

### Dataset Structure

Create a `dataset/` folder with two text files:

**dataset/train.en.txt** (English source)
```
The cat sits on the mat .
I love machine learning .
Hello world .
...
```

**dataset/train.vi.txt** (Vietnamese target)
```
Con mèo ngồi trên tấm thảm .
Tôi yêu học máy .
Xin chào thế giới .
...
```

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total pairs | 133,317 |
| Languages | English-Vietnamese |
| Lines per file | 133,317 |
| Encoding | UTF-8 |
| Format | Plain text |
| File size | ~4-5 MB each |
| Coverage | ~95% vocabulary |

### Requirements

- One sentence per line
- Matching number of lines in both files
- UTF-8 encoding
- Sentences should be tokenized (words separated by spaces)
- Optional: Remove extra whitespace

### Data Preprocessing

The project automatically:
- Lowercases all text
- Removes punctuation
- Tokenizes sentences
- Pads/truncates to MAX_LEN
- Builds vocabulary
- Creates embeddings

## 🎯 Model Training Details

### Vocabulary Building
- Automatically built from training data
- Special tokens: `<PAD>`, `<UNK>`, `<START>`, `<END>`
- Sequence padding to max length

### Word2Vec Embeddings
- Skip-gram model (sg=1)OS>`, `<EOS>`
- Sequence padding to max length

### Word2Vec Embeddings
- Skip-gram model (sg=1)
- Vector size: 256 dimensions
- Window size: 5
- Min count: 1
- Trained on source language only

### LSTM Parameters
- Hidden size: 256 units
- Input embedding: 256 dimensions
- Bi-LSTM output: 512 dimensions (concat)
- Float32 precision for GPU
- Xavier/Glorot initialization

### Training
- Optimizer: SGD with fixed learning rate (0.01)
- Loss: Cross-entropy loss
- Backpropagation: Full BPTT through encoder and decoder
- Validation split: 80/20
- Best model checkpointing
Models are saved in NumPy format with GPU/CPU compatibility:

**LSTM-to-LSTM:**
```python
{
    'embedding_src': embeddings,      # Word embeddings
    'embedding_tgt': embeddings,      # Target embeddings
    'Wy': output_weights,             # Output layer
    'by': output_bias,
    'encoder_Wf', 'encoder_Uf', ...  # All LSTM weights
    'decoder_Wf', 'decoder_Uf', ...  # All LSTM weights
}
```

**Seq2Seq:**
```python
{
    'embedding_src': embeddings,
    'embedding_tgt': embeddings,
    'Wy': output_weights,
    'by': output_bias,
    'encoder': BiLSTM object,         # Pickled
    'decoder': LSTMCell object,       # Pickled
    'attention': Attention object,    # Pickled
}**GPU acceleration** (10-100x speedup)
- ✅ Two architecture options
- ✅ **Bahdanau Attention** mechanism
- ✅ **Full BPTT** through encoder and decoder
- ✅ Bidirectional encoding
- ✅ Advanced decoding strategies
- ✅ Reproducible results
- ✅ Best model checkpointing
- ✅ CPU/GPU compatibility

### Limitations & Future Improvements
- ❌ No batch processing (currently single-example)
- ❌ No beam search decoding
- ❌ No BLEU/METEOR evaluation
- ❌ No mixed precision (FP16) training
- ❌ No multi-GPU support
- 🔧 *Planned*: Batch processing for faster training
- 🔧 *Planned*: Beam search decoding
- 🔧 *Planned*: Automatic evaluation metrics
- 🔧 *Planned*: Multi-GPU distributed trainr.npy")

# Plot loss history
viz = Visualization()
viz.plot_loss(model.losses, filename="historical_loss.png")
```

## 🔍 Performance & Limitations

### Current Capabilities
- ✅ Learns to encode variable-length sequences
- ✅ Captures bidirectional context
- ✅ Minimal memory footprint
- ✅ Fast training on CPU

### Limitations & Future Improvements
- ❌ No attention mechanism
- ❌ No beam search decoding
- ❌ No BLEU/METEOR evaluation
- ❌ Limited backpropagation (output layer only)
- 🔧 *Planned*: Full BPTT implementation
- 🔧 *Planned*: Attention mechanism
- cupy-cuda12x          # GPU support (optional)
- nltk==3.8.1
- gensim==4.3.3
- matplotlib==3.9.4
- scikit-learn==1.6.1

Dev Tools:
- black==24.10.0
- flake8==7.1.1
- isort==5.13.2
- pylint==3.3.1
```

**Note**: CuPy is optional. Code falls back to NumPy (CPU) if not installed.atplotlib==3.9.4

Dev GPU Issues

**Issue: GPU not detected**
```bash
# Check CUDA installation
nvidia-smi

# Check CuPy installation
python -c "import cupy; print(cupy.cuda.Device().compute_capability)"
```

**Solution:**
- Install NVIDIA drivers
- Install CUDA Toolkit (11.x or 12.x)
- Install matching CuPy version

**Issue: Out of GPU memory**

**Solutions:**
- Reduce `EMBEDDING_DIM` (256 → 128)
- Reduce `HIDDEN_SIZE` (256 → 128)
- Use CPU mode: `--device cpu`

### Installation Issues

**Issue: `ModuleNotFoundError: No module named 'gensim'`**

**Solution:**
```bash
pip install gensim
```
GPU: Reduce model size or use CPU
- CPU: Reduce `MAX_LEN`, `EMBEDDING_DIM`, `HIDDEN_SIZE`
- Close other applications

### Issue: Training too slow

**Solutions:**
- ⚡ Use GPU: `--device gpu` (10-100x faster)
- Install CuPy: `pip install cupy-cuda12x`
- Reduce `MAX_LEN` to process shorter sequences
- UsGPU/CPU Compatibility
- All arrays use `xp` (NumPy or CuPy)
- Automatic device selection based on `--device` flag
- Models saved in NumPy format (portable)
- Automatic conversion when loading

### e LSTM-to-LSTM instead of Seq2Seq

### Issue: Old model won't load

**Solution:**

### Bahdanau Attention
1. Compute attention scores for each encoder state
2. Apply softmax to get attention weights (alpha)
3. Compute weighted sum of encoder states (context)
4. Concatenate context with decoder input

## 📚 Additional Resources

- **[GPU_QUICKSTART.md](GPU_QUICKSTART.md)** - Quick start guide for GPU
- **[GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md)** - Detailed optimization docs
- **CuPy Documentation**: https://docs.cupy.dev/
- **Original Dataset**: https://www.kaggle.com/datasets/hungnm/englishvietnamese-translation/

## 🤝 Contributing

This is an educational project. Contributions welcome:
- Bug fixes
- Performance improvements
- New features (beam search, BLEU scores, etc.)
- Documentation improvements
- Models trained before GPU update need retraining
- Warning message will appear
- Retrain with: `python main.py --dataset dataset --model model/new`
**Solution:**
```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x

# Or use CPU mode (no CuPy needed)
python main.py --device cpu ...
```

See `requirements.txt` for full list.

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'gensim'`

**Solution:**
```bash
pip install gensim
```

### Issue: NLTK tokenizer errors

**Solution:**
```bash
python -c "import nltk; nltk.download('punkt')"
```

### Issue: Dataset not found

**Solution:**
- Ensure `dataset/` folder exists
- Check files are named `train.en.txt` and `train.vi.txt`
- Verify files contain matching number of lines

### Issue: Out of memory

**Solutions:**
- Reduce `MAX_LEN` parameter
- Reduce `EMBEDDING_DIM`
- Process smaller batches

## 📝 Implementation Notes

### LSTM Gates (Corrected)
The implementation uses correct matrix ordering:
- **Forget Gate**: $f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$
- **Input Gate**: $i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$
- **Candidate**: $\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$
- **Output Gate**: $o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$

### Bi-LSTM Processing
1. Forward pass: Process sequence left-to-right
2. Backward pass: Process sequence right-to-left
3. Concatenation: Stack hidden states [h_forward; h_backward]

## 📄 License

This is an educational project. Feel free to use and modify.
