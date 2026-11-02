# Emo_Text — Phased Implementation Plan

This plan modernizes Emo_Text incrementally while **preserving the original graduation project**. The original scripts, models concept, and Tkinter GUI remain in the repository as the academic artifact; changes are additive or clearly scoped refactors. Four kinds of work are distinguished:

1. **Refactoring existing code** — repairing the current scripts without changing models.
2. **Improving the original ML/DL pipeline** — keeping BERT + GAN concepts, making them runnable and correct.
3. **Introducing modern alternative models** — new models (RoBERTa, Fish Speech/CSM, etc.) alongside or replacing originals, with trade-offs documented.
4. **Building a new production-ready version** — FastAPI + web UI + Docker, separate from the graduation artifact.

## Guiding principles

- Do not delete or overwrite the original files (`script.py`, `test_surprise.py`, `ops.py`, `utils.py`, GUI variants).
- Do not claim accuracy or quality metrics that are not measured in the repo.
- Each phase ends in a runnable, committed state.
- Hardware assumptions: a developer laptop with optional NVIDIA GPU (CUDA) for training/TTS inference.

---

## Phase 0 — Foundation (weeks 1–2). Goal: cloneable, runnable repo

| # | Task | Kind | Complexity | Hardware | Backward compatible |
|---|------|------|------------|----------|---------------------|
| 0.1 | Add `requirements.txt` with pinned versions (transformers, torch, PyPDF2/pypdf, gTTS, playsound, easyocr, Pillow, librosa, soundfile) | Refactor | Low | None | Yes |
| 0.2 | Add `.gitignore` (venv, `__pycache__`, `.idea`, checkpoints, logs, `.ipynb_checkpoints`) | Refactor | Low | None | Yes |
| 0.3 | Fix `pdf_to_text.py`: extract all pages, context manager, return text instead of print-only | Refactor | Low | None | Yes (text output unchanged) |
| 0.4 | Replace hardcoded `C:/Users/Michael/...` paths with `__file__`-relative paths; missing-asset fallback | Refactor | Low | None | Yes |
| 0.5 | Fix Tkinter popups: use `Toplevel`, single root window | Refactor | Low | None | Yes |
| 0.6 | Add `README.md` (done), `LICENSE` (choose, e.g., MIT), contribution guidelines | Docs | Low | None | Yes |
| 0.7 | Verify end-to-end: `python add_image_gui.py` launches and classifies/types text (internet needed for HF Hub) | Verification | Low | None | Yes |

**Exit criteria:** `git clone && pip install -r requirements.txt && python add_image_gui.py` works on a clean machine.

---

## Phase 1 — Quality and reliability (weeks 3–5). Goal: testable, observable prototype

| # | Task | Kind | Complexity | Hardware | Backward compatible |
|---|------|------|------------|----------|---------------------|
| 1.1 | Add `tests/` with pytest: emotion label mapping, text chunking, PDF all-page extraction, gTTS folder creation | New tests | Medium | None | Yes |
| 1.2 | GitHub Actions workflow: run tests on Python 3.10/3.11/3.12 on push and PR | CI | Low | None | Yes |
| 1.3 | Add `logging` and `try/except` around file I/O, HF download, playback; `config.yaml` for paths/model IDs/labels | Refactor | Medium | None | Yes |
| 1.4 | Make BERT inference robust: singleton model loader, confidence threshold, GPU/CPU device selection, batching | Improve pipeline | Medium | CPU/GPU | Yes |
| 1.5 | Non-blocking GUI: run classification/TTS in a background thread; remove `time.sleep(1)` freeze | Refactor | Medium | None | Yes |
| 1.6 | Consolidate `gui.py`/`gui_new.py`/`add_image_gui.py` into a single maintained entry point (keep others archived or removed with notice) | Refactor | Medium | No (entry-point change) | Partially |

**Exit criteria:** 80%+ test coverage of core modules; GUI no longer freezes; all paths configurable.

---

## Phase 2 — Make the ML pipeline correct and measurable (weeks 6–9). Goal: verifiable accuracy

| # | Task | Kind | Complexity | Hardware | Backward compatible |
|---|------|------|------------|----------|---------------------|
| 2.1 | Fine-tune BERT/RoBERTa on an emotion dataset; record accuracy, confusion matrix, per-class F1 | Improve pipeline | Medium | GPU (train), CPU (infer) | No (new weights) |
| 2.2 | Add `training/train_bert.py` script with train/val split, early stopping, MLflow/W&B tracking | New training | Medium | GPU | Yes (additive) |
| 2.3 | Publish fine-tuned weights (HF Hub or local `models/`) and point `script.py` at them with offline fallback | Improve pipeline | Medium | None | Yes |
| 2.4 | Migrate EmoMUNIT GAN from TF 1.x (`tf.contrib`, `tensorflow_addons`) to PyTorch/Keras 3 **or** retire it; remove module-level `gan.test()` on import | Improve pipeline | High | GPU (train) | No (model code changes) |
| 2.5 | Add evaluation script for GAN (if kept): reconstruction error, qualitative samples | New evaluation | Medium | GPU | Yes (additive) |

**Exit criteria:** documented BERT accuracy with code that produces it; GAN either runnable on modern TF/PyTorch or explicitly retired with rationale.

---

## Phase 3 — Expressive audio (weeks 10–14). Goal: audible emotion

| # | Task | Kind | Complexity | Hardware | Backward compatible |
|---|------|------|------------|----------|---------------------|
| 3.1 | Wrap gTTS behind an interface so a modern TTS can be swapped in without changing the GUI | Refactor | Low | None | Yes |
| 3.2 | Integrate one modern zero-shot expressive TTS (Fish Speech / Sesame CSM / Chatterbox / EmoVoice) with emotion conditioning and optional voice cloning | Modern alternative | High | GPU | Interface-compatible |
| 3.3 | If spectrogram-based TTS chosen, add neural vocoder (HiFi-GAN / BigVGAN / VOCOS) | Modern alternative | Medium | GPU | Yes |
| 3.4 | Compare gTTS vs. new TTS qualitatively (samples stored in `samples/`) | Evaluation | Medium | GPU | Yes |

**Exit criteria:** generated audio audibly reflects the selected emotion; voice-cloning demo works from a short reference clip.

---

## Phase 4 — Production version (weeks 15–20). Goal: deployable system

| # | Task | Kind | Complexity | Hardware | Backward compatible |
|---|------|------|------------|----------|---------------------|
| 4.1 | FastAPI backend: `/extract`, `/classify`, `/synthesize`, `/jobs` endpoints; Pydantic schemas | New production | Medium | None | Additive |
| 4.2 | Celery + Redis workers for long synthesis jobs; job polling from frontend | New production | Medium | None | Additive |
| 4.3 | Web frontend: Streamlit (fastest) or Next.js + React (custom UI) | New production | Medium–High | None | Additive |
| 4.4 | PDF hardening: PyMuPDF layout/TOC/chapter detection; OCR fallback for scans | Improve pipeline | High | CPU (GPU optional for OCR) | Yes |
| 4.5 | Multilingual emotion classification + TTS (XLM-RoBERTa, multilingual TTS) | Modern alternative | High | GPU (train) | No (UI changes) |
| 4.6 | Docker (CPU + CUDA images), ONNX Runtime export + INT8 quantization for BERT/TTS | New production | High | GPU for build | Yes |
| 4.7 | GitHub Actions CI/CD: tests, lint, build images | CI | Medium | None | Yes |

**Exit criteria:** one-command `docker compose up` deployment; async job API; web UI usable from a browser.

---

## Trade-offs: original vs. modern models

| Aspect | Original (kept) | Modern alternative | Trade-off |
|--------|----------------|--------------------|-----------|
| Emotion model | BERT base (`friends-bert-base-uncased`) | RoBERTa / DistilBERT fine-tuned | Original: simple, HF Hub-hosted, accuracy unverified. Modern: better accuracy, multi-label, requires fine-tuning dataset + GPU training. |
| TTS | gTTS (not emotional) | Fish Speech / CSM / Chatterbox / EmoVoice | Original: zero-setup, robotic, no emotion. Modern: expressive, speaker-consistent, needs GPU and license review. |
| Voice conversion | EmoMUNIT GAN (TF 1.x, untrained) | Retire or migrate to PyTorch | Original: academic value, non-runnable on TF 2.x. Modern: runnable but requires re-training and new data. |
| UI | Tkinter desktop | Streamlit / Next.js | Original: runs locally, hardcoded paths. Modern: accessible, deployable, requires frontend work. |

## Suggested first sprint

1. `requirements.txt` + `.gitignore`
2. Fix `pdf_to_text.py` all-page extraction
3. Fix hardcoded paths + Tkinter `Toplevel`
4. Verify `add_image_gui.py` runs end to end

These five items make the repository cloneable and runnable without changing any model behavior, and they unblock every later phase.
