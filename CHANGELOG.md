# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-10-19

### Added

- `--host` and `--port` command-line options.
- `ARCFACE_MIN_FACE_SIZE` and `ARCFACE_SAVE_EVERY_N_FILES` environment variables, so these values no longer require editing `face_recognizer.py`.
- Thai README as a separate file, `README.th.md`.
- Documentation for GPU setup, configuration, API usage, troubleshooting, known limitations and responsible use.

### Changed

- README restructured, with the changelog moved to this file.
- Model license terms stated explicitly: InsightFace pretrained models are for non-commercial research use only.

### Fixed

- The language link in the README pointed to the README itself.

### Security

- The server now binds to `127.0.0.1` by default instead of `0.0.0.0`. To access the web interface from another machine, start the server with `--host 0.0.0.0`.

## [1.0.0] - 2026-10-19

### Added

- Face detection and 512-dimensional ArcFace embeddings with InsightFace `buffalo_l`, using CUDA with automatic CPU fallback.
- Sorting of photos and videos into `Sorted/Person_XXXX/`, with `Waiting/` for borderline matches and files without faces.
- Video support: frames sampled from the start of each clip, with repeated faces in a clip merged into one person.
- Persistent person database and processed-file index, so runs can be stopped and resumed.
- English web interface with live progress, settings, a person list with thumbnails, rename and merge.
- HTTP API for processing control, persons and settings.
- Windows helper scripts `INSTALL.bat` and `START.bat`.
- Documentation in English and Thai.
