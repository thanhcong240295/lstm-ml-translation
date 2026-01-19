# Machine Learning Translation - Seq2Seq with Bi-LSTM

A sequence-to-sequence machine translation model using Bidirectional LSTM with Word2Vec embeddings, trained from scratch without high-level deep learning frameworks.

## 📋 Project Overview

This project implements an English-Vietnamese neural machine translation (NMT) system using:
- **Bidirectional LSTM (Bi-LSTM)** for encoding sequences
- **Word2Vec embeddings** for word representations
- **Custom LSTM cell implementation** from numpy
- **Minimal backpropagation** through output layer

The model is designed to learn phrase-level translation patterns through self-attention via LSTM gates.

## ✨ Features

- ✅ Custom LSTM implementation (no TensorFlow/PyTorch)
- ✅ Bidirectional encoding for context understanding
- ✅ Word2Vec embedding training on dataset
- ✅ Automatic loss visualization and statistics
- ✅ Model checkpointing with loss history
- ✅ Preprocessing pipeline with tokenization
- ✅ Code quality tools (black, isort, flake8, pylint)
- ✅ Clean, modular architecture

## 📁 Project Structure

```
machinelearning-translation/
├── main.py                    # Entry point
├── language_translation.py    # Translation model (Seq2Seq trainer)
├── bi_lstm.py                 # Bidirectional LSTM implementation
├── lstm.py                    # LSTM cell implementation
├── word2vec.py                # Word2Vec embedding trainer
├── vocab.py                   # Vocabulary management
├── preprocessor.py            # Data preprocessing & tokenization
├── utils.py                   # Utility functions for CLI
├── visualization.py           # Loss plotting & metrics visualization
├── check_code.py              # Code quality checker
├── format_code.py             # Code formatter (black + isort)
├── clean.py                   # Project cleanup script
├── dataset/                   # Training data
│   ├── train.en.txt          # English source sentences
│   └── train.vi.txt          # Vietnamese target sentences
├── model/                     # Trained model weights
├── result/                    # Training outputs
│   ├── training_loss.png
│   ├── loss_statistics.png
│   └── model.npy
└── requirements.txt           # Python dependencies
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

3. **Download NLTK data (for tokenization):**
```bash
python -c "import nltk; nltk.download('punkt')"
```

### Usage

#### Training the Model

```bash
python main.py --dataset "dataset" --model "model/translator.npy"
```

**Parameters:**
- `--dataset` : Path to dataset directory (containing .txt files)
- `--model` : Path to save trained model

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
- **Output Gate**: Decides what to output from cell state

### Key Components

| Module | Purpose | Key Classes |
|--------|---------|------------|
| `lstm.py` | LSTM cell implementation | `LSTMCell` |
| `bi_lstm.py` | Bidirectional wrapper | `BiLSTM` |
| `language_translation.py` | Training & inference | `LanguageTranslation` |
| `word2vec.py` | Embedding training | `Word2VecEmbedding` |
| `vocab.py` | Vocabulary management | `Vocab` |
| `preprocessor.py` | Data preprocessing | `Preprocessor` |
| `visualization.py` | Results visualization | `Visualization` |

## 📈 Training Output

During training, the model displays:
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
- Skip-gram model (sg=1)
- Vector size: 256 dimensions
- Window size: 5
- Min count: 1

### LSTM Parameters
- Hidden size: 256 units
- Forward + Backward LSTM
- Concatenated output: 512 dimensions
- Output layer: vocab_size × 512

### Training
- Optimizer: Basic SGD with fixed learning rate
- Loss: Cross-entropy (negative log likelihood)
- Batch processing: Sentence-by-sentence
- No gradient clipping (improvements possible)

## 💾 Model Checkpoints

Saved model includes:
```python
{
    'Wy': output_layer_weights,    # (vocab_size, 512)
    'by': output_layer_bias,       # (vocab_size, 1)
    'input_size': 256,
    'hidden_size': 256,
    'losses': [8.234, 7.125, ...]  # Loss per epoch
}
```

To load and visualize saved model:

```python
from language_translation import LanguageTranslation
from visualization import Visualization

model = LanguageTranslation()
model.load_model("model/translator.npy")

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
- 🔧 *Planned*: Beam search decoding

## 📦 Dependencies

```
Core ML:
- numpy==1.26.4
- nltk==3.8.1
- gensim==4.3.3
- matplotlib==3.9.4

Dev Tools:
- black==24.10.0
- flake8==7.1.1
- isort==5.13.2
- pylint==3.3.1
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
