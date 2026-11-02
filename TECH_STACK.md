# Emo_Text — Recommended Technology Stack (2026)

This stack modernizes the graduation project while keeping the scope realistic for an individual developer or small team. Every entry includes the rationale and the hardware/licensing implications. Sources: official project documentation and public model repositories (Hugging Face, GitHub) consulted via web search.

## Recommended stack

| Layer | Recommendation | Version (indicative) | Why |
|-------|---------------|----------------------|-----|
| Language | Python | 3.10–3.12 | Widely supported by ML libraries; matches current env (3.11.7) |
| Backend / API | FastAPI | latest stable | Async REST, automatic OpenAPI docs, Pydantic validation; industry standard for ML serving |
| Background jobs | Celery + Redis | Celery 5.x, Redis 7.x | Proven async task queue for long-running synthesis jobs |
| Frontend (web) | Streamlit (option A) or Next.js + React (option B) | Streamlit 1.x / Next.js 15+ | Streamlit: fastest path from Python, no JS required; Next.js: full web app if a custom UI is justified |
| Desktop (legacy) | Tkinter | stdlib | Kept as-is for the original local GUI; no new investment |
| Emotion classification | transformers + torch | transformers 4.x, torch 2.x | RoBERTa / DistilBERT fine-tuned on emotion data |
| Expressive TTS | Fish Audio S2 / Fish Speech, Sesame CSM, Resemble Chatterbox, or EmoVoice | model-dependent | 2025–2026 zero-shot expressive TTS with emotion control and voice cloning |
| Neural vocoder (optional) | HiFi-GAN, BigVGAN, or VOCOS | model-dependent | If a spectrogram-based TTS is chosen |
| PDF extraction | PyMuPDF (fitz) + pypdf | latest stable | Robust text/layout/TOC extraction; OCR fallback with easyOCR or Tesseract |
| OCR | easyOCR or Tesseract | latest stable | Scanned-document support; easyOCR already referenced in `ocr.py` |
| Experiment tracking | MLflow or Weights & Biases | latest stable | Run tracking and model registry for reproducibility |
| Inference optimization | ONNX Runtime + quantization | onnxruntime 1.x | CPU-friendly inference; INT8 quantization reduces latency/memory |
| Containerization | Docker | latest | CPU and GPU image variants; reproducible environments |
| CI/CD | GitHub Actions | — | Tests, linting, build on push/PR |
| Testing | pytest | 7.x+ | Unit and integration tests |
| Linting / formatting | ruff + black | latest | Fast linting and formatting |
| Type checking | mypy | latest | Gradual type annotations |

## Notes on specific recommendations

### Emotion classification
The current code uses `friends-bert-base-uncased` from Hugging Face Hub (accuracy unverified; no training code). A modern replacement would fine-tune `cardiffnlp/twitter-roberta-base-emotion-latest` or `SamLowe/roberta-base-go_emotions` on a labeled dataset, with multi-label support and confidence thresholds. Hardware: CPU is sufficient for inference; GPU recommended for fine-tuning.

### Expressive TTS
gTTS (currently used) is not controllable for emotion — the `emotion` argument only names a folder. Modern alternatives (active, recent public repositories):
- **Fish Audio S2 / Fish Speech** — recent zero-shot TTS with emotion and speaker control (`fishaudio/fish-speech`).
- **Sesame CSM** (Conversational Speech Model) — `SesameAILabs/csm` (2025).
- **Resemble Chatterbox** — `resemble-ai/chatterbox` on Hugging Face, expressive TTS with voice cloning.
- **EmoVoice** — LLM-based emotional TTS with text prompting (ACM MM 2025).

Review each license (MIT/Apache-2.0 vs non-commercial) before productizing. GPU strongly recommended for inference.

### Neural vocoder
If the TTS pipeline produces mel-spectrograms, pair with **HiFi-GAN**, **BigVGAN**, or **VOCOS** for waveform synthesis (PyTorch/ONNX export available).

### PDF extraction
Replace `PyPDF2.PdfFileReader` (deprecated; last-page-only bug in current code) with **PyMuPDF** (fitz) for text, layout, table of contents, and chapter detection, plus **pypdf** as a lightweight alternative. Add OCR (easyOCR/Tesseract) for scanned/image-only pages.

### Experiment tracking
Use **MLflow** (self-hostable, includes model registry) or **Weights & Biases** (SaaS, free tier) to track BERT fine-tuning runs and any migrated GAN experiments.

### Deployment
- **Docker:** multi-stage Dockerfiles, CPU base image and CUDA variant.
- **ONNX Runtime:** export BERT (and smaller TTS components) to ONNX; apply dynamic quantization for CPU deployment.
- **GPU acceleration:** CUDA-enabled worker images for TTS inference.

### What to avoid
- **TensorFlow 1.x / tf.contrib / tensorflow_addons:** discontinued; incompatible with TF 2.16+. The current `ops.py` cannot run on TF 2.21. Migrate to PyTorch/Keras 3 or retire the GAN prototype.
- **PyPDF2 `PdfFileReader`/`getPage`:** deprecated API; use `pypdf.PdfReader` or PyMuPDF.
- **gTTS for emotional audio:** acceptable as a placeholder, but not emotional by design.

## Licensing summary (verify before use)

- FastAPI, Celery, Redis, pytest, ruff, mypy, Docker, ONNX Runtime: permissive (MIT/Apache-2.0).
- PyMuPDF: check `pymupdf` PyPI license (AGPL considerations for the SDK); pypdf is BSD-style.
- easyOCR: Apache-2.0.
- TTS models (Fish Speech, CSM, Chatterbox, EmoVoice): model-specific; many carry non-commercial or research-only restrictions. Confirm on Hugging Face model cards before public deployment.
- MLflow: Apache-2.0; Weights & Biases: proprietary SaaS (free tier available).

