# Emo_Text — Modernization Roadmap

This roadmap converts the graduation-project prototype into a maintainable, extensible system. Every item distinguishes **verified facts** (from the code review in `CODE_REVIEW.md`) from **recommendations** (assumptions marked as such). Complexity, hardware needs, and backward compatibility are given per item.

## How to read

- **Quick wins** — small, immediate-impact changes; complexity Low.
- **High-impact improvements** — architecture, quality, usability; complexity Medium–High.
- **Advanced features** — research-oriented additions; complexity High.

Backward compatibility: **"Compatible"** means existing behavior/interfaces are preserved; **"Breaking"** means interfaces change; **"N/A"** means no existing interface exists yet.

---

## Phase A — Quick wins (complexity: Low)

### A1. Add `requirements.txt` and pin versions
**What:** Create `requirements.txt` with pinned versions for all imports: `transformers`, `torch`, `PyPDF2`/`pypdf`, `gTTS`, `playsound`, `easyocr`, `Pillow`, `librosa`, `soundfile`, plus TensorFlow only if the GAN is repaired.
**Why:** The repo currently cannot be installed reproducibly; most imports fail in a clean environment (verified: PyPDF2, gTTS, easyocr, PyAudio, playsound, librosa, soundfile, tensorflow_addons, transformers all missing/broken here).
**Benefit:** New contributors can `pip install -r requirements.txt` and run the app.
**Complexity:** Low
**Hardware:** None
**Backward compatible:** Yes

### A2. Add `.gitignore` and stop committing large binaries
**What:** Add a standard Python `.gitignore` (venv, `__pycache__`, `.idea`, checkpoints, logs, `.ipynb_checkpoints`). Do **not** delete the already-committed `side_img.jpg` (2 MB) or `doc_model_one.docx` (620 KB) — just stop adding new large binaries; consider Git LFS for future assets.
**Why:** Repository hygiene; avoids bloating git history with model checkpoints and caches later.
**Complexity:** Low
**Hardware:** None
**Backward compatible:** Yes

### A3. Fix the PDF extraction bug (last page only)
**What:** In `pdf_to_text.py`, extract text from **all pages** (loop `range(numPages)` or use `pypdf.PdfReader`), close the file handle with a context manager, and return concatenated text instead of only printing it.
**Why:** Verified bug — `getPage(x - 1)` extracts only the final page; `PdfFileReader`/`getPage` are deprecated in modern PyPDF2/pypdf.
**Benefit:** Core PDF→text feature actually works for multi-page documents.
**Complexity:** Low
**Hardware:** None
**Backward compatible:** Yes (output type unchanged)

### A4. Fix hardcoded GUI asset paths
**What:** Replace `C:/Users/Michael/OneDrive/Desktop/gp/img0.png` (and similar) in `gui.py`/`add_image_gui.py` with paths relative to the script file (`os.path.join(os.path.dirname(__file__), 'img0.png')`), and guard missing assets with a fallback (text-only button).
**Why:** Verified portability failure — the GUI only works on the author's machine.
**Complexity:** Low
**Hardware:** None
**Backward compatible:** Yes

### A5. Consolidate Tkinter popup logic
**What:** Replace every `tk.Tk()` used for popups with `tk.Toplevel`; keep one root window.
**Why:** `alert_popup` creates a second root window — incorrect Tkinter usage that can hang or misbehave.
**Complexity:** Low
**Hardware:** None
**Backward compatible:** Yes

## Phase B — High-impact improvements (complexity: Medium–High)

### B1. Add unit and integration tests (pytest)
**What:** Add `tests/` with pytest: emotion-label mapping, text chunking, PDF text concatenation, gTTS folder creation, WAV loading. Add `pytest` and GitHub Actions workflow running tests on Python 3.10/3.11/3.12.
**Why:** Zero test coverage exists (`test_surprise.py` is a GAN script, not a test). pytest 7.4.0 is available in the environment.
**Benefit:** Prevents regressions during refactoring; documents expected behavior.
**Complexity:** Medium
**Hardware:** None
**Backward compatible:** Yes

### B2. Repair or replace the TF 1.x GAN pipeline
**What:** Either (a) migrate `ops.py`/`test_surprise.py` from `tf.contrib` / TensorFlow Addons to Keras 3 / `tf.keras`, or (b) replace the EmoMUNIT prototype with a PyTorch implementation of a modern emotion voice-conversion baseline. Remove module-level `gan.test()` execution on import.
**Why:** Verified incompatibility: installed TensorFlow 2.21.0 has no `tf.contrib` and TensorFlow Addons is discontinued/incompatible with TF 2.16+.
**Benefit:** The experimental GAN becomes runnable and importable without side effects.
**Complexity:** High
**Hardware:** GPU recommended for training; CPU for inference
**Backward compatible:** Breaking (model code changes; checkpoints not loadable across migration)

### B3. Make emotion classification robust and observable
**What:** Add confidence thresholding, fallback to "neutral" below threshold, batching, lazy model loading (singleton), GPU/CPU device selection, and structured logging. Wrap HF Hub download in try/except with a clear offline error message.
**Why:** `script.py` instantiates a new model per prediction, has no confidence handling, and fails silently offline.
**Complexity:** Medium
**Hardware:** CPU sufficient for inference; GPU preferred
**Backward compatible:** Yes (label set unchanged)


### B4. Make TTS actually emotional
**What:** Replace/inject a controllable emotional TTS behind `generate_audio.py` (see Phase C for specific models), keeping the same `generate(text, emotion)` interface where possible.
**Why:** Verified gap: gTTS ignores the `emotion` argument (only names a subfolder); output is not emotionally modulated.
**Complexity:** Medium–High
**Hardware:** GPU for neural TTS inference (CPU possible but slow)
**Backward compatible:** Interface-compatible if wrapped

### B5. Add error handling, logging, and configuration
**What:** `try/except` around file I/O, model download, and playback; a `config.yaml`/`.env` for paths, model IDs, emotion labels, sample rate; logging instead of `print`.
**Why:** No error handling exists anywhere; paths and hyperparameters are hardcoded.
**Complexity:** Medium
**Hardware:** None
**Backward compatible:** Yes

### B6. Non-blocking UI with background synthesis
**What:** Run classification and TTS in a background thread/queue; show a "generating…" state; avoid `time.sleep(1)`.
**Why:** `audio.generate()` blocks the Tkinter main thread and freezes the app.
**Complexity:** Medium
**Hardware:** None
**Backward compatible:** Yes


## Phase C — Advanced features (complexity: High)

### C1. Replace gTTS with state-of-the-art expressive TTS
**What:** Integrate a modern zero-shot emotional TTS: Fish Audio S2 / Fish Speech, Sesame CSM, Resemble Chatterbox, or EmoVoice (LLM-based emotional TTS with text prompting). Support voice cloning/speaker consistency from a short reference clip.
**Why:** 2025–2026 models produce far more natural, controllable emotional prosody than gTTS (see `TECH_STACK.md`).
**Benefit:** Emotion actually audible in output; multilingual support.
**Complexity:** High
**Hardware:** GPU strongly recommended (CUDA)
**Backward compatible:** Breaking (new inference service)

### C2. Modernize emotion classification
**What:** Fine-tune a RoBERTa/distilBERT checkpoint (e.g., `cardiffnlp/twitter-roberta-base-emotion-latest` or `SamLowe/roberta-base-go_emotions`) on a target dataset; add multi-label support and confidence scores.
**Why:** More current than the BERT base checkpoint; GoEmotions gives finer-grained emotion taxonomies.
**Complexity:** Medium
**Hardware:** GPU for training; CPU for inference
**Backward compatible:** Breaking if label set changes

### C3. Multilingual support
**What:** Multilingual emotion classification (XLM-RoBERTa) + multilingual TTS; OCR language selection beyond English.
**Why:** Current OCR is English-only; gTTS supports many languages but with no emotion control.
**Complexity:** High
**Hardware:** GPU for training/fine-tuning
**Backward compatible:** Breaking (UI language selection needed)

### C4. Web frontend (Next.js/React or Streamlit) + FastAPI backend
**What:** Replace Tkinter with a browser UI; FastAPI backend with async endpoints (`/extract`, `/classify`, `/synthesize`), Celery/RQ + Redis background jobs for long synthesis.
**Why:** Desktop Tkinter limits distribution; a web UI is shareable and easier to maintain.
**Benefit:** Accessible from any device; asynchronous long-running jobs.
**Complexity:** High
**Hardware:** Server with GPU for synthesis workers
**Backward compatible:** Breaking (replaces desktop GUI)

### C5. PDF hardening: OCR for scans, layout, chapters
**What:** PyMuPDF (fitz) for text + layout + TOC/chapter detection; OCR fallback (Tesseract or easyOCR) for scanned pages; preserve headings for per-chapter audiobooks.
**Why:** PyPDF2 struggles with scanned/image-only PDFs; no layout/chapter awareness exists.
**Complexity:** High
**Hardware:** CPU; GPU optional for OCR
**Backward compatible:** Yes (additive)

### C6. Experiment tracking and model registry
**What:** MLflow or Weights & Biases for training runs (BERT fine-tuning, GAN); versioned model artifacts.
**Why:** No tracking exists; essential for academic rigor and reproducibility.
**Complexity:** Medium
**Hardware:** None
**Backward compatible:** Yes

### C7. Deployment packaging
**What:** Dockerfile (CPU and GPU variants), GitHub Actions CI/CD, ONNX Runtime export for BERT/TTS inference, quantization (INT8) for CPU deployment.
**Why:** No containerization or serving story; ONNX/quantization materially improves inference cost.
**Complexity:** High
**Hardware:** GPU for building GPU images
**Backward compatible:** Yes (additive packaging)

## Summary matrix

| Item | Phase | Complexity | Hardware | Backward compatible | Primary benefit |
|------|-------|------------|----------|---------------------|-----------------|
| requirements.txt | A1 | Low | None | Yes | Reproducible installs |
| .gitignore / LFS | A2 | Low | None | Yes | Clean repo |
| Fix PDF last-page bug | A3 | Low | None | Yes | Core feature works |
| Fix hardcoded paths | A4 | Low | None | Yes | Runs on any machine |
| Toplevel popups | A5 | Low | None | Yes | Stable GUI |
| Tests + CI | B1 | Medium | None | Yes | Regressions caught |
| Migrate/replace GAN | B2 | High | GPU (train) | Breaking | Runnable GAN |
| Robust BERT inference | B3 | Medium | CPU/GPU | Yes | Reliable classification |
| Emotional TTS | B4/C1 | Medium–High | GPU | Interface-compatible | Audible emotion |
| Error handling + config | B5 | Medium | None | Yes | Operability |
| Non-blocking UI | B6 | Medium | None | Yes | Usability |
| SOTA TTS / voice cloning | C1 | High | GPU | Breaking | Quality |
| Modern emotion models | C2 | Medium | GPU (train) | Breaking | Better accuracy |
| Multilingual | C3 | High | GPU (train) | Breaking | Reach |
| Web UI + FastAPI | C4 | High | GPU (server) | Breaking | Accessibility |
| PDF OCR + chapters | C5 | High | CPU | Yes | Robust input |
| Experiment tracking | C6 | Medium | None | Yes | Reproducibility |
| Docker + ONNX + quant | C7 | High | GPU (build) | Yes | Deployment |

**Suggested first sprint:** A1 → A2 → A3 → A4 → A5. These make the repository cloneable and runnable without changing any model behavior.
