<div align="center">

# $\color{#1F6FEB}\textsf{ArcFace Dataset Sorter}$

**Sort large photo and video collections into per-person folders with ArcFace face recognition, entirely on your own machine.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#requirements)
[![Model](https://img.shields.io/badge/model-InsightFace%20buffalo__l-8250df)](#license)
[![Runs locally](https://img.shields.io/badge/runs-100%25%20local-2da44e)](#security-and-privacy)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

**English** · [ภาษาไทย](README.th.md)

</div>

---

[Overview](#overview) · [Features](#features) · [Requirements](#requirements) · [Quick Start](#quick-start) · [Usage](#usage) · [Configuration](#configuration) · [HTTP API](#http-api) · [How It Works](#how-it-works) · [Security and Privacy](#security-and-privacy) · [Responsible Use](#responsible-use) · [Known Limitations](#known-limitations) · [Troubleshooting](#troubleshooting) · [Changelog](#changelog) · [License](#license) · [Credits](#credits)

---

<a id="overview"></a>

## $\color{#1F6FEB}\textsf{Overview}$

ArcFace Dataset Sorter groups a photo and video collection by *who appears in it*. It detects faces, compares them with a person database that grows as it runs, and copies each file into that person's folder. Uncertain matches are set aside for review instead of being guessed.

![Web Interface](<ArcFace-Based Face Dataset Organizer — v1.0.0/docs/web-ui.png>)

It is designed for curated collections of roughly 1–2 TB in which faces are reasonably clear, with someone reviewing the `Waiting/` folder after each run. The project is a processing pipeline and web interface built on a pre-trained model; it does not train or distribute a model of its own.

| Good fit | Not a good fit |
|---|---|
| Curated, cleaned datasets | CCTV or surveillance footage |
| Clear faces | Very small or blurry faces |
| Batch processing | Real-time processing |
| Manual review of uncertain results | Identity verification or authentication |

<a id="features"></a>

## $\color{#1F6FEB}\textsf{Features}$

- **Fully offline** — photos, videos and face data never leave your machine.
- **Photos and videos** — videos are sampled from the start of each clip, and repeated faces within a clip count as one person.
- **Three-way decision** — confident matches are sorted, borderline matches go to review, and unknown faces become new persons.
- **Non-destructive** — source files are copied, never moved or deleted.
- **Resumable** — the person database and processed-file index persist, so a stopped run continues where it left off.
- **Web interface** — live progress, settings, a person list with thumbnails, rename and merge.
- **HTTP API** — start runs, read progress and manage persons from scripts.
- **GPU acceleration** — CUDA through ONNX Runtime, with automatic fallback to CPU.

**Supported formats**

| Type | Extensions |
|---|---|
| Images | JPG, JPEG, PNG, BMP, WEBP, GIF |
| Videos | MP4, AVI, MOV, MKV, WMV, FLV, WEBM |

<a id="requirements"></a>

## $\color{#1F6FEB}\textsf{Requirements}$

| Component | Requirement |
|---|---|
| Operating system | Windows (primary platform). Linux and macOS are supported on a best-effort basis; see [Known Limitations](#known-limitations). |
| Python | 3.10 or newer (`INSTALL.bat` installs 3.11) |
| Python packages | Listed in [`requirements.txt`](requirements.txt) |
| GPU | Optional. NVIDIA GPU with CUDA; see [GPU acceleration](#gpu-acceleration-optional). |
| Disk space | About as much free space as the data being processed, because files are copied |
| Network | Required on first start only, to download the model |

<a id="quick-start"></a>

## $\color{#1F6FEB}\textsf{Quick Start}$

<a id="windows"></a>

### $\color{#8250DF}\textsf{Windows}$

1. Run `INSTALL.bat`. It installs Python 3.11 with winget if Python is missing, then installs the dependencies.
2. Run `START.bat`.
3. Open <http://localhost:5000>.

<a id="linux--macos"></a>

### $\color{#8250DF}\textsf{Linux / macOS}$

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open <http://localhost:5000>.

On first start, the InsightFace `buffalo_l` model is downloaded automatically.

<a id="gpu-acceleration-optional"></a>

### $\color{#8250DF}\textsf{GPU acceleration (optional)}$

ONNX Runtime is distributed as two separate packages: `onnxruntime` (CPU) and `onnxruntime-gpu` (CUDA). Install only one of them. To enable the GPU:

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

Install CUDA and cuDNN versions that match your ONNX Runtime release, as listed in the [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements). Installing or upgrading InsightFace can reinstall `onnxruntime`; repeat the swap afterwards.

<a id="usage"></a>

## $\color{#1F6FEB}\textsf{Usage}$

1. Copy curated photos and videos directly into `Clean/`. Subfolders are not scanned.
2. Click **Start** in the web interface and follow the progress.
3. Review the results:

```text
Clean/                        Input, left unchanged
Sorted/
├── Person_0001/              One folder per person (rename or merge in the web interface)
└── Person_0002/
Waiting/
├── possibly_Person_0001/     Borderline matches to review
└── no_face_detected/         Files in which no face was found
```

Clicking **Stop** and then **Start** again resumes from where processing stopped.

> [!TIP]
> Before a large run, process a small sample and check `Sorted/` and `Waiting/` to tune the thresholds for your data.

<a id="configuration"></a>

## $\color{#1F6FEB}\textsf{Configuration}$

<a id="command-line-options"></a>

### $\color{#8250DF}\textsf{Command-line options}$

| Option | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Address to bind to. Use `0.0.0.0` to allow access from other machines (see [Security and Privacy](#security-and-privacy)). |
| `--port` | `5000` | Port to listen on. |

```bash
python app.py --host 0.0.0.0 --port 5050
```

On Windows, add the options to the `python app.py` line in `START.bat`.

<a id="processing-settings"></a>

### $\color{#8250DF}\textsf{Processing settings}$

Change these in the **Settings** tab or with `POST /api/settings`. Changes take effect from the next file processed.

Both thresholds are **cosine distances** between face embeddings: 0 means identical, and lower values are stricter.

![Distance thresholds](<ArcFace-Based Face Dataset Organizer — v1.0.0/docs/thresholds.png>)

| Setting | Default | Recommended | Description |
|---|---|---|---|
| Similarity Threshold | 0.40 | 0.35–0.45 | A face within this distance of a known person is a match. |
| Waiting Threshold | 0.48 | — | Files between the two thresholds go to `Waiting/`; faces beyond this distance become a new person. |
| Min Face Confidence | 0.70 | — | Minimum detector confidence for a face to be used. |
| Video Scan Length | 20 s | 10–20 s | Only this opening part of each video is scanned. |
| Video Frame Interval | 2.0 s | 1–3 s | Time between sampled frames. With the defaults, about 10 frames per clip. |

Settings apply only to files processed after the change. To re-evaluate files that were already processed, clear the processed-file index with `POST /api/reset-processed` and start a new run.

<a id="environment-variables"></a>

### $\color{#8250DF}\textsf{Environment variables}$

| Variable | Default | Description |
|---|---|---|
| `ARCFACE_MIN_FACE_SIZE` | `50` | Minimum face size in pixels. Smaller faces are ignored. |
| `ARCFACE_SAVE_EVERY_N_FILES` | `10` | How often, in files, the database is written to disk. |

```bash
# Linux / macOS
ARCFACE_MIN_FACE_SIZE=40 python app.py

# Windows (Command Prompt)
set ARCFACE_MIN_FACE_SIZE=40
python app.py
```

<a id="http-api"></a>

## $\color{#1F6FEB}\textsf{HTTP API}$

The web interface is built on the same API. Base URL: `http://localhost:5000`.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/start` | Start processing |
| `POST` | `/api/stop` | Stop processing |
| `GET` | `/api/status` | Progress, statistics and recent log entries |
| `GET` | `/api/stats/summary` | Summary counts |
| `GET` | `/api/persons` | List persons |
| `POST` | `/api/persons/<id>/rename` | Rename a person |
| `POST` | `/api/persons/merge` | Merge two persons |
| `GET` | `/api/thumbnail/<id>` | Face thumbnail of a person |
| `GET` | `/api/folders` | Folder listing |
| `GET` | `/api/settings` | Read settings |
| `POST` | `/api/settings` | Update settings |
| `POST` | `/api/reset-processed` | Clear the processed-file index |

```bash
# Start a run and check progress
curl -X POST http://localhost:5000/api/start
curl http://localhost:5000/api/status

# Read the current settings
curl http://localhost:5000/api/settings
```

<a id="how-it-works"></a>

## $\color{#1F6FEB}\textsf{How It Works}$

![Processing pipeline](<ArcFace-Based Face Dataset Organizer — v1.0.0/docs/pipeline.png>)

1. **Scan** — reads files at the top level of `Clean/` and skips those already in the processed-file index.
2. **Detect and embed** — InsightFace detects each face and computes a 512-dimensional ArcFace embedding.
3. **Match** — each embedding is compared with the person database by cosine distance:
   - within the Similarity Threshold → `Sorted/Person_XXXX/`
   - between the two thresholds → `Waiting/possibly_Person_XXXX/`
   - beyond the Waiting Threshold → a new person is created
   - no face detected → `Waiting/no_face_detected/`
4. **Videos** — frames are sampled at a fixed interval from the start of the clip, and repeated faces within the clip are merged into one person.
5. **Copy** — the file is copied to its destination; the source is never modified.

| Aspect | Detail |
|---|---|
| Model | InsightFace `buffalo_l` (face detection and ArcFace recognition) |
| Inference | ONNX Runtime; CUDA when available, otherwise CPU |
| Processing | Sequential, one file at a time |
| Video cost | About 10 decoded frames per clip with the default settings, regardless of clip length |
| Person database | Up to 20 embeddings per person (about 40 KB) |
| State | `.face_db/` (person database, processed-file index) and `.thumbnails/`, written to disk every 10 files |
| Logs | `face_recognition.log` and the console |

<a id="security-and-privacy"></a>

## $\color{#1F6FEB}\textsf{Security and Privacy}$

- **Local by default.** The server binds to `127.0.0.1` and is reachable only from this computer.
- **No authentication.** Anyone who can reach the server can view thumbnails, rename and merge persons, and start or stop runs. Use `--host 0.0.0.0` only on a trusted network, and never expose the port to the internet.
- **No outbound data.** The only network access is the one-time model download.
- **Biometric data.** `.face_db/` and `.thumbnails/` contain face embeddings and face images. Both are excluded in `.gitignore`; do not publish or share them. Deleting both folders removes the person database and thumbnails.

<a id="responsible-use"></a>

## $\color{#1F6FEB}\textsf{Responsible Use}$

Face recognition processes biometric data, which many laws treat as sensitive personal data, including Thailand's Personal Data Protection Act (PDPA) and the EU General Data Protection Regulation (GDPR). Use this tool only on collections you are entitled to process, with a lawful basis such as the consent of the people pictured. It is not intended for surveillance, tracking individuals, or identity verification.

<a id="known-limitations"></a>

## $\color{#1F6FEB}\textsf{Known Limitations}$

- Only the top level of `Clean/` is scanned; subfolders are ignored.
- Files are processed one at a time.
- Only the opening part of each video is scanned, so people who appear later in a clip are not detected.
- The **Open folder** buttons use Windows Explorer and work only on Windows.
- Accuracy drops for small, blurred, occluded or extreme-angle faces; such files tend to end up in `Waiting/`.

<a id="troubleshooting"></a>

## $\color{#1F6FEB}\textsf{Troubleshooting}$

| Symptom | Solution |
|---|---|
| Processing runs on the CPU although an NVIDIA GPU is installed | The CPU package `onnxruntime` is installed, alone or alongside `onnxruntime-gpu`. Follow [GPU acceleration](#gpu-acceleration-optional) and check that the CUDA and cuDNN versions match. |
| `pip install` fails on Windows with "Microsoft Visual C++ 14.0 or greater is required" | Install Microsoft C++ Build Tools with the *Desktop development with C++* workload, then run `INSTALL.bat` again. |
| The model download fails or is interrupted | Check the internet connection, delete the incomplete `buffalo_l` folder under `~/.insightface/models/` (InsightFace's default location) and start again. |
| Port 5000 is already in use | Start on another port, for example `python app.py --port 5050`. |
| Too many files end up in `Waiting/` | Raise the Similarity Threshold slightly (for example 0.42–0.45). |
| Different people are sorted into the same person | Lower the Similarity Threshold (for example 0.35–0.38), reset the processed-file index and run again. |
| One person is split across several persons | Use **Merge** in the web interface. Raising the Similarity Threshold slightly reduces splits. |

<a id="changelog"></a>

## $\color{#1F6FEB}\textsf{Changelog}$

See [CHANGELOG.md](CHANGELOG.md).

<a id="license"></a>

## $\color{#1F6FEB}\textsf{License}$

The source code in this repository is released under the [MIT License](LICENSE).

**The MIT License does not cover the model weights.** The InsightFace pretrained models used by this project, including `buffalo_l`, which is downloaded automatically on first start, are licensed by InsightFace for **non-commercial research use only**. For commercial use, obtain a license from InsightFace or replace the model with one you are licensed to use. See the [InsightFace license](https://github.com/deepinsight/insightface#license) for the current terms.

<a id="credits"></a>

## $\color{#1F6FEB}\textsf{Credits}$

- **[InsightFace](https://github.com/deepinsight/insightface)** — library and the `buffalo_l` (ArcFace) model used for detection and embeddings
- **ArcFace** — Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019
- [ONNX Runtime](https://onnxruntime.ai/) · [OpenCV](https://opencv.org/) · [NumPy](https://numpy.org/) · [Flask](https://flask.palletsprojects.com/) · [Pillow](https://python-pillow.org/)

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{deng2019arcface,
  title     = {ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author    = {Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {4690--4699},
  year      = {2019}
}
```

</details>
