# Emo_Text

Emotionally expressive audiobook generation from PDF documents — a Python desktop application that extracts text, classifies its emotion with BERT, and reads it aloud.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-1.x%2F2.x-orange.svg)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green.svg)

> **Status:** graduation-project prototype. See [Limitations](#limitations-and-future-improvements) and the [modernization roadmap](./MODERNIZATION_ROADMAP.md) before using in production.

## Overview

**Emo_Text** converts documents into audio with emotion-aware playback. A user loads a PDF (or an image via OCR) or types a sentence; the system detects the emotion of the text (neutral, joy, sadness, fear, anger, surprise, disgust) using a BERT classifier and speaks it back through a Tkinter desktop application.

A secondary, experimental voice-conversion pipeline (TensorFlow GAN, `EmoMUNIT`) was prototyped to transform neutral speech into a target emotional style, but it is **not integrated** into the running application and ships without trained weights.

## Key features

- **PDF → text** extraction (PyPDF2; note: current implementation extracts the last page only — see [Known issues](#known-issues)).
- **Image → text** via OCR (easyOCR, English).
- **Emotion classification** with a fine-tuned BERT model (`friends-bert-base-uncased`): 7 classes — neutral, joy, sadness, fear, anger, surprise, disgust.
- **Text-to-speech playback** with gTTS (Google Text-to-Speech), sentence by sentence.
- **Desktop GUI** built with Tkinter (file browser, emotion display, play button).
- **Experimental GAN voice conversion** (`EmoMUNIT`): WORLD-vocoder MCEP features, style/content encoders, LSGAN adversarial training, cycle-consistency and reconstruction losses. Prototype only — no checkpoints included.

## System architecture

```mermaid
flowchart TD
    subgraph Input
        PDF[PDF file] --> PT[pdf_to_text.py]
        IMG[Image / scan] --> OC[ocr.py]
        TXT[Typed text]
    end
    PT --> T[Text]
    OC --> T
    TXT --> T
    T --> SC[script.py<br/>BERT emotion classifier]
    SC --> EM[Emotion label]
    EM --> GUI[Tkinter GUI<br/>add_image_gui.py]
    EM --> GA[generate_audio.py<br/>gTTS MP3 per sentence]
    GA --> SP[(Speakers)]
    GUI --> SP

    subgraph Experimental (not wired into GUI)
        WAV[WAV audio] --> WU[utils.py<br/>WORLD vocoder<br/>MCEP features]
        WU --> GAN[test_surprise.py / ops.py<br/>EmoMUNIT GAN<br/>neutral → surprise]
        GAN --> OUT[Converted WAV]
    end
```

> The solid flow is what the GUI actually executes. The dashed experimental GAN path is a standalone prototype with no shipped weights or training data.

## Technologies and libraries

| Layer | Library | Purpose |
|-------|---------|---------|
| Language | Python 3.11 | Implementation language |
| GUI | Tkinter + Pillow | Desktop application and images |
| NLP | Hugging Face `transformers` (BERT) | Emotion classification |
| PDF | PyPDF2 | Text extraction from PDF |
| OCR | easyOCR | Image → text (English) |
| TTS | gTTS, playsound | Speech synthesis and playback |
| Audio DSP | librosa, PyWorld, soundfile | WORLD vocoder feature extraction/synthesis (GAN pipeline) |
| Deep learning | TensorFlow 1.x-style code, TensorFlow Addons | EmoMUNIT GAN prototype |
| Visualization | Matplotlib, TensorBoard | Training diagnostics |

## Installation and environment setup

**Important:** the repository has no `requirements.txt`, and most dependencies are **not installed** in a clean environment. The application will not run until they are installed.

### 1. Clone the repository

```bash
git clone https://github.com/MichaelMansour256/Emo_Text.git
cd Emo_Text
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

Install the runtime dependencies manually (no `requirements.txt` ships with the repo):

```bash
pip install transformers torch PyPDF2 gTTS playsound easyocr Pillow librosa soundfile
```

The GAN pipeline additionally requires **TensorFlow** and **TensorFlow Addons** in TF 1.x-compatible versions:

```bash
# Note: TensorFlow Addons is discontinued and incompatible with TF 2.16+.
# The GAN code (ops.py, test_surprise.py) was written for TF 1.x and
# must be migrated before it can run on modern TensorFlow.
pip install tensorflow tensorflow-addons
```

### 4. Model weights

The BERT classifier downloads `friends-bert-base-uncased` from Hugging Face Hub on first run — an internet connection is required. No local model weights or GAN checkpoints are included in the repository.

### 5. Run the application

The most complete GUI entry point is:

```bash
python add_image_gui.py
```

1. Click **Browse A File** and select a PDF, JPG, or PNG.
2. Click **get the text** — extracted text appears in the editor.
3. Click the play button — each sentence is classified by emotion and spoken with gTTS.

**Expected blockers out of the box:** missing packages (see dependency table above), hardcoded asset paths in `gui.py` (`C:/Users/Michael/OneDrive/Desktop/gp/...`), and gTTS requiring internet access.

## Dataset information and preprocessing

- **No dataset is included in the repository.**
- PDF extraction (`pdf_to_text.py`) reads the final page only and performs no cleaning or sentence segmentation.
- OCR (`ocr.py`) is hardcoded to English with GPU disabled.
- The GAN pipeline expects folders `neutral/` and `surprise/` containing WAV files (`train_A_dir` / `train_B_dir`) and `test_neutral/` for validation. These directories do not exist in the repo.
- To make the project reproducible, you will need: (a) a labeled emotion text dataset for fine-tuning/validating the BERT classifier, and (b) a paired emotional speech dataset for the GAN voice-conversion experiment.

## Model architecture and methodology

### Emotion classification — `script.py`

- **Model:** `BertForSequenceClassification` from `friends-bert-base-uncased` (BERT base, uncased).
- **Classes:** neutral, joy, sadness, fear, anger, surprise, disgust.
- **Inference:** tokenization (truncation/padding to 512 tokens) → forward pass → softmax → `argmax` label.
- **Training:** none in the repository. Accuracy figures (e.g., ~89%) cited in project materials are **not verifiable** from the source code.

### Experimental emotional voice conversion — `test_surprise.py`, `ops.py`, `utils.py`

- **Architecture:** MUNIT-inspired `EmoMUNIT` GAN with separate style and content encoders, a shared decoder, and two discriminators.
- **Audio representation:** WORLD-vocoder MCEP features (24 coefficients, 16 kHz, 5 ms frames); pitch handled separately via log-f0 mean/std conversion between domains.
- **Losses:** LSGAN adversarial, L1 reconstruction, style consistency, content consistency, cycle consistency.
- **Direction:** neutral (domain A) → surprise (domain B), style dimension 16.
- **Status:** prototype. Defined and configurable, but not trained in this repo, not integrated with the GUI, and incompatible with installed TensorFlow 2.21 (`tf.contrib` / `tensorflow_addons` removed).

### What actually speaks

The running GUI uses **gTTS**, which converts text to MP3 via Google's TTS API. The `emotion` label is only used to name the output subfolder — gTTS does not support emotional prosody, so the audible output is not emotion-modulated.

## Training and evaluation instructions

**There are no training scripts and no evaluation scripts in this repository.**

- The BERT model must be fine-tuned externally (e.g., on an emotion-labeled dataset such as GoEmotions or an equivalent) and uploaded to Hugging Face Hub under `friends-bert-base-uncased`, or the `model_path` in `script.py` must be changed to a local directory.
- The GAN training loop lives in `EmoMUNIT.train()` (`test_surprise.py`) and expects WAV folders `neutral/` and `surprise/`. It writes TensorBoard logs to `logs/` and checkpoints to `checkpoint/`, but no training data or checkpoints are provided.


## Inference and application usage

1. Start the app: `python add_image_gui.py`.
2. Load a PDF or image via **Browse A File**; typed text can also be used directly.
3. Extract text with **get the text**.
4. Press the emotion/play button: each comma-separated segment is classified and spoken in turn.
5. Emotion labels are shown on screen (`emotion is: <label>`).

Known operational issues:
- `gui.py` references images at `C:/Users/Michael/OneDrive/Desktop/gp/img0.png` — fails on other machines.
- `generate_audio.py` writes to `gtts/<emotion>/` without creating the folder.
- Predictions block the UI; no threading or async playback.

## Screenshots and demo

| Asset | File | Notes |
|-------|------|-------|
| App logo / title card | `text.png` | 600 × 200 PNG |
| GUI background | `side_img.jpg` | 3024 × 4032 JPEG |
| Play button | `img0.png` | 284 × 108 PNG |
| Small icon | `text2.png` | 220 × 320 PNG |

**Missing:** actual running-application screenshots, demo audio samples, and architecture screenshots. The `.docx` documentation (`doc_model_one.docx`) exists but its text layer could not be reliably extracted.

## Project structure

```
Emo_Text/
├── README.md
├── CODE_REVIEW.md
├── MODERNIZATION_ROADMAP.md
├── MODERN_ARCHITECTURE.md
├── TECH_STACK.md
├── IMPLEMENTATION_PLAN.md
├── MISSING_ASSETS.md
├── add_image_gui.py    # Most complete GUI entry point
├── gui.py              # Tkinter GUI variant (hardcoded paths)
├── gui_new.py          # Tkinter GUI variant (structurally broken)
├── script.py           # BERT emotion inference
├── pdf_to_text.py      # PDF extraction (last page only)
├── ocr.py              # easyOCR stub
├── generate_audio.py   # gTTS playback
├── test_surprise.py    # EmoMUNIT GAN model + train/test (TF 1.x)
├── ops.py              # TF layers/losses (TF 1.x contrib APIs)
├── utils.py            # WORLD vocoder DSP + data sampling
├── 1.pdf               # Sample PDF
├── 1.txt               # Sample text (author's CV)
├── doc_model_one.docx  # Project documentation
├── img0.png            # Button asset
├── side_img.jpg        # Background asset
├── text.png            # Logo asset
└── text2.png           # Icon asset
```

**Not present:** `requirements.txt`, `tests/`, `models/`, `data/`, `checkpoint/`, `.gitignore`, `LICENSE`.

## Results and performance metrics

- **No quantitative results are recorded in the repository.** No training logs, confusion matrices, or audio quality metrics (MOS, PESQ, STOI) exist.
- The ~89% BERT accuracy mentioned in project materials is **not supported by any code or logs** in this repo and should be treated as unverified.
- No inference latency, memory usage, or audio-quality benchmarks have been measured.

## Limitations and future improvements

- **No reproducible environment** (missing `requirements.txt`; most packages absent).
- **TF 1.x GAN code incompatible** with modern TensorFlow (`tf.contrib`, `tensorflow_addons` removed).
- **No trained weights** for BERT fine-tuning or the GAN; gTTS is not emotional.
- **PDF extraction is broken** for multi-page documents (last page only).
- **Hardcoded absolute paths** prevent the GUI from running on other machines.
- **No tests, no CI/CD, no error handling, no logging.**
- **Single-threaded UI** freezes during synthesis.

See `MODERNIZATION_ROADMAP.md` for a prioritized, complexity-graded improvement plan.

## Contributing

This is a graduation project. Contributions are welcome in the form of bug reports, dependency pinning, test coverage, and the modernization items in `IMPLEMENTATION_PLAN.md`. Please open an issue to discuss larger changes before submitting pull requests.

## License

No `LICENSE` file is present in the repository. If you intend to reuse or distribute this code, add a license (e.g., MIT) and a `.gitignore` for Python environments.
