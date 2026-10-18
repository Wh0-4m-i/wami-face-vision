"""
ArcFace Dataset Sorter - Backend Engine
Uses InsightFace (ArcFace) for high-accuracy face detection and recognition
Groups similar faces into folders automatically
"""

import os
import json
import shutil
import logging
import tempfile
import threading
import traceback
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('face_recognition.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Constants ---
BASE_DIR = Path(__file__).parent
CLEAN_DIR = BASE_DIR / "Clean"
OUTPUT_DIR = BASE_DIR / "Sorted"
WAITING_DIR = BASE_DIR / "Waiting"
FACE_DB_DIR = BASE_DIR / ".face_db"
THUMBNAILS_DIR = BASE_DIR / ".thumbnails"

# Face recognition thresholds
# NOTE: these are read at runtime (not bound as default args) so that
# changes made through /api/settings take effect on the next file.
SIMILARITY_THRESHOLD = 0.40   # cosine distance threshold (lower = more strict)
MIN_CONFIDENCE = 0.70          # minimum face detection confidence
VIDEO_SCAN_SECONDS = 20.0      # only scan the first N seconds of each video
VIDEO_FRAME_INTERVAL = 2.0     # sample one frame every N seconds (within the scan window)
MIN_FACE_SIZE = 50             # minimum face width/height in pixels
WAITING_THRESHOLD = 0.48       # above this distance goes to Waiting folder
SAVE_EVERY_N_FILES = 10        # checkpoint database to disk every N files

SUPPORTED_IMAGES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
SUPPORTED_VIDEOS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}


def normalize(vec):
    """L2-normalize an embedding (float32)"""
    vec = np.asarray(vec, dtype=np.float32)
    return vec / (np.linalg.norm(vec) + 1e-10)


class FaceDatabase:
    """Manages known face embeddings and their cluster assignments"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.index_file = db_path / "face_index.json"
        self.processed_file = db_path / "processed_files.json"

        self.person_embeddings = {}   # person_id -> list of normalized embeddings
        self.person_metadata = {}     # person_id -> {name, count, thumbnail}
        self.processed = {}           # filename -> {size, mtime, status, ...}
        self.next_id = 1

        # Cached matrix of all embeddings for vectorized matching
        self._matrix = None
        self._matrix_ids = None
        self._dirty = True

        self.lock = threading.RLock()
        self._load()

    # ─── Persistence ─────────────────────────────────────────────────────────

    def _load(self):
        """Load existing database"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.next_id = data.get('next_id', 1)
                # Convert string keys to int
                self.person_metadata = {int(k): v for k, v in data.get('persons', {}).items()}
                logger.info(f"Loaded face database: {len(self.person_metadata)} persons")
            except Exception as e:
                logger.error(f"Error loading database: {e}")

        # Load embeddings
        emb_dir = self.db_path / "embeddings"
        emb_dir.mkdir(exist_ok=True)

        for person_id in self.person_metadata:
            emb_file = emb_dir / f"person_{person_id}.npy"
            if emb_file.exists():
                self.person_embeddings[person_id] = [normalize(e) for e in np.load(str(emb_file))]

        # Load list of already-processed files
        if self.processed_file.exists():
            try:
                with open(self.processed_file, 'r', encoding='utf-8') as f:
                    self.processed = json.load(f)
                logger.info(f"Loaded processed-file index: {len(self.processed)} files")
            except Exception as e:
                logger.error(f"Error loading processed-file index: {e}")

        self._dirty = True

    @staticmethod
    def _atomic_write_json(path: Path, data):
        tmp = path.with_suffix(path.suffix + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def save(self):
        """Save database to disk (atomic writes so a crash can't corrupt it)"""
        with self.lock:
            try:
                self._atomic_write_json(self.index_file, {
                    'persons': {str(k): v for k, v in self.person_metadata.items()},
                    'next_id': self.next_id
                })
                self._atomic_write_json(self.processed_file, self.processed)

                emb_dir = self.db_path / "embeddings"
                emb_dir.mkdir(exist_ok=True)
                for person_id, embeddings in self.person_embeddings.items():
                    if embeddings:
                        np.save(str(emb_dir / f"person_{person_id}.npy"), np.array(embeddings))

                # Remove embedding files of persons that no longer exist (e.g. merged)
                for emb_file in emb_dir.glob("person_*.npy"):
                    try:
                        pid = int(emb_file.stem.split('_')[1])
                    except (IndexError, ValueError):
                        continue
                    if pid not in self.person_metadata:
                        emb_file.unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Error saving database: {e}")

    # ─── Processed-file tracking ─────────────────────────────────────────────

    @staticmethod
    def _file_signature(path: Path):
        st = path.stat()
        return st.st_size, int(st.st_mtime)

    def is_processed(self, path: Path):
        rec = self.processed.get(path.name)
        if not rec:
            return False
        try:
            size, mtime = self._file_signature(path)
        except OSError:
            return False
        return rec.get('size') == size and rec.get('mtime') == mtime

    def mark_processed(self, path: Path, status, persons=None):
        with self.lock:
            size, mtime = self._file_signature(path)
            self.processed[path.name] = {
                'size': size,
                'mtime': mtime,
                'status': status,
                'persons': persons or [],
                'processed_at': datetime.now().isoformat(timespec='seconds')
            }

    def clear_processed(self):
        with self.lock:
            self.processed = {}
            self.save()

    # ─── Matching ────────────────────────────────────────────────────────────

    def _rebuild_matrix(self):
        ids, vecs = [], []
        for pid, embs in self.person_embeddings.items():
            for e in embs:
                ids.append(pid)
                vecs.append(e)
        if vecs:
            self._matrix = np.vstack(vecs).astype(np.float32)
            self._matrix_ids = np.array(ids)
        else:
            self._matrix = None
            self._matrix_ids = None
        self._dirty = False

    def find_best_match(self, embedding):
        """Find the best matching person. Returns (person_id, distance)"""
        with self.lock:
            if self._dirty:
                self._rebuild_matrix()
            if self._matrix is None:
                return None, 1.0

            sims = self._matrix @ normalize(embedding)
            idx = int(np.argmax(sims))
            return int(self._matrix_ids[idx]), 1.0 - float(sims[idx])

    def add_person(self, embedding, thumbnail_path=None):
        """Create a new person entry"""
        with self.lock:
            person_id = self.next_id
            self.next_id += 1

            self.person_embeddings[person_id] = [normalize(embedding)]
            self.person_metadata[person_id] = {
                'name': f'Person_{person_id:04d}',
                'count': 1,
                'thumbnail': thumbnail_path,
                'created_at': datetime.now().isoformat()
            }
            self._dirty = True
            return person_id

    def update_person(self, person_id, embedding, max_samples=20):
        """Add a new embedding sample to existing person"""
        with self.lock:
            embeddings = self.person_embeddings.setdefault(person_id, [])
            embeddings.append(normalize(embedding))

            if len(embeddings) > max_samples:
                self.person_embeddings[person_id] = embeddings[-max_samples:]

            if person_id in self.person_metadata:
                self.person_metadata[person_id]['count'] = \
                    self.person_metadata[person_id].get('count', 0) + 1
            self._dirty = True

    def rename_person(self, person_id, new_name):
        with self.lock:
            if person_id in self.person_metadata:
                self.person_metadata[person_id]['name'] = new_name
                self.save()

    def merge(self, from_id, to_id):
        with self.lock:
            from_embs = self.person_embeddings.pop(from_id, [])
            self.person_embeddings.setdefault(to_id, []).extend(from_embs)
            from_meta = self.person_metadata.pop(from_id, {})
            if to_id in self.person_metadata:
                self.person_metadata[to_id]['count'] = \
                    self.person_metadata[to_id].get('count', 0) + from_meta.get('count', 0)
            self._dirty = True
            self.save()

    def get_all_persons(self):
        with self.lock:
            items = list(self.person_metadata.items())
            emb_counts = {pid: len(e) for pid, e in self.person_embeddings.items()}

        result = {}
        for pid, meta in items:
            folder = OUTPUT_DIR / meta['name']
            file_count = sum(1 for _ in folder.iterdir()) if folder.exists() else 0
            result[str(pid)] = {
                **meta,
                'id': pid,
                'file_count': file_count,
                'embedding_count': emb_counts.get(pid, 0)
            }
        return result


class FaceRecognitionEngine:
    """Main face recognition engine"""

    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.is_running = False
        self.should_stop = False
        self.stats = self._empty_stats()

        self.face_analyzer = None
        self.db = FaceDatabase(FACE_DB_DIR)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        WAITING_DIR.mkdir(parents=True, exist_ok=True)
        THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _empty_stats(status='idle'):
        return {
            'total_files': 0,
            'processed_files': 0,
            'skipped_files': 0,
            'current_file': '',
            'faces_detected': 0,
            'persons_found': 0,
            'waiting_count': 0,
            'status': status,
            'errors': [],
            'log': []
        }

    def reset_stats(self, status='idle'):
        self.stats = self._empty_stats(status)

    def log(self, message, level='info'):
        timestamp = datetime.now().strftime('%H:%M:%S')
        entry = {'time': timestamp, 'level': level, 'message': message}
        self.stats['log'].append(entry)

        if len(self.stats['log']) > 200:
            self.stats['log'] = self.stats['log'][-200:]

        if level == 'error':
            logger.error(message)
            self.stats['errors'].append(message)
        else:
            logger.info(message)

        if self.progress_callback:
            self.progress_callback(self.stats)

    def initialize_model(self):
        if self.face_analyzer is not None:
            return True  # already loaded from a previous run
        try:
            self.log("Loading InsightFace ArcFace model...")
            from insightface.app import FaceAnalysis

            self.face_analyzer = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            self.log("Model loaded successfully (buffalo_l ArcFace)")
            return True
        except Exception as e:
            self.log(f"Error loading InsightFace: {e}", 'error')
            return False

    # ─── Detection ───────────────────────────────────────────────────────────

    @staticmethod
    def _read_image(image_path):
        """Read an image; works with non-ASCII (e.g. Thai) Windows paths and GIFs"""
        try:
            data = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass
        pil_img = Image.open(image_path).convert('RGB')  # first frame for GIFs
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _valid_faces(self, img):
        """Run detector and keep faces passing confidence / size filters"""
        result = []
        for face in self.face_analyzer.get(img):
            if face.det_score < MIN_CONFIDENCE or face.embedding is None:
                continue
            x1, y1, x2, y2 = face.bbox.astype(int)
            if (x2 - x1) >= MIN_FACE_SIZE and (y2 - y1) >= MIN_FACE_SIZE:
                result.append(face)
        return result

    def get_faces_from_image(self, image_path):
        """Returns list of (embedding, face, img)"""
        try:
            img = self._read_image(image_path)
            if img is None:
                return []
            return [(normalize(f.embedding), f, img) for f in self._valid_faces(img)]
        except Exception as e:
            self.log(f"Error processing image {Path(image_path).name}: {e}", 'error')
            return []

    @staticmethod
    def _open_video(video_path):
        """Open a video; falls back to a temp ASCII copy for non-ASCII paths"""
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            return cap, None
        cap.release()
        if str(video_path).isascii():
            return None, None
        tmp_dir = tempfile.mkdtemp(prefix='facerec_')
        tmp_path = os.path.join(tmp_dir, 'video' + Path(video_path).suffix)
        shutil.copy2(video_path, tmp_path)
        cap = cv2.VideoCapture(tmp_path)
        if cap.isOpened():
            return cap, tmp_dir
        cap.release()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, None

    def get_faces_from_video(self, video_path):
        """
        Sample frames from the first VIDEO_SCAN_SECONDS seconds only and
        cluster faces within the video so the same person counts once.
        Returns list of (mean_embedding, best_face, best_frame),
        or None if processing was stopped by the user.
        """
        cap, tmp_dir = self._open_video(video_path)
        if cap is None:
            self.log(f"Cannot open video: {Path(video_path).name}", 'error')
            return []

        clusters = []  # each: {'sum', 'face', 'frame', 'score'}
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            duration = frame_count / fps if frame_count > 0 else None

            interval = max(0.25, float(VIDEO_FRAME_INTERVAL))
            scan_end = float(VIDEO_SCAN_SECONDS)
            if duration:
                scan_end = min(scan_end, duration)

            # Timestamps centred in each interval (skips the often-black first frame)
            timestamps = list(np.arange(interval / 2, scan_end, interval))
            if not timestamps:
                timestamps = [scan_end / 2]

            for t in timestamps:
                if self.should_stop:
                    return None

                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
                ret, frame = cap.read()
                if not ret:
                    continue

                try:
                    faces = self._valid_faces(frame)
                except Exception:
                    continue

                for face in faces:
                    emb = normalize(face.embedding)
                    best_c, best_dist = None, float('inf')
                    for c in clusters:
                        dist = 1.0 - float(np.dot(emb, normalize(c['sum'])))
                        if dist < best_dist:
                            best_c, best_dist = c, dist

                    if best_c is not None and best_dist <= SIMILARITY_THRESHOLD:
                        best_c['sum'] = best_c['sum'] + emb
                        if face.det_score > best_c['score']:
                            best_c.update(face=face, frame=frame, score=float(face.det_score))
                    else:
                        clusters.append({'sum': emb.copy(), 'face': face,
                                         'frame': frame, 'score': float(face.det_score)})
        except Exception as e:
            self.log(f"Error processing video {Path(video_path).name}: {e}", 'error')
        finally:
            cap.release()
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        return [(normalize(c['sum']), c['face'], c['frame']) for c in clusters]

    # ─── Matching & sorting ──────────────────────────────────────────────────

    def save_face_thumbnail(self, face, img, person_id):
        try:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            h, w = img.shape[:2]
            pad = int(max(x2 - x1, y2 - y1) * 0.3)
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0:
                return None

            face_crop = cv2.resize(face_crop, (128, 128))
            thumb_path = THUMBNAILS_DIR / f"person_{person_id:04d}.jpg"
            ok, buf = cv2.imencode('.jpg', face_crop)
            if ok:
                buf.tofile(str(thumb_path))  # unicode-safe write
                return str(thumb_path)
            return None
        except Exception:
            return None

    def process_face(self, embedding, face, img):
        """
        Match one face against the database.
        Returns (person_id, kind, distance) where kind is 'matched' | 'waiting' | 'new'.
        Does NOT copy files; the caller copies once per file.
        """
        person_id, distance = self.db.find_best_match(embedding)

        if person_id is not None and distance <= SIMILARITY_THRESHOLD:
            self.db.update_person(person_id, embedding)
            return person_id, 'matched', distance

        if person_id is not None and distance <= WAITING_THRESHOLD:
            return person_id, 'waiting', distance

        with self.db.lock:
            thumb = self.save_face_thumbnail(face, img, self.db.next_id)
            person_id = self.db.add_person(embedding, thumb)
        self.log(f"  New person detected: {self.db.person_metadata[person_id]['name']}")
        return person_id, 'new', distance

    @staticmethod
    def _copy_into(src: Path, dest_dir: Path):
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)

    def copy_to_waiting(self, source_file, possible_person_id):
        src_path = Path(source_file)
        meta = self.db.person_metadata.get(possible_person_id)
        if meta:
            waiting_sub = WAITING_DIR / f"possibly_{meta['name']}"
        else:
            waiting_sub = WAITING_DIR / "unknown"
        self._copy_into(src_path, waiting_sub)

    def sort_file(self, src: Path, faces):
        """Match all faces of a file and copy it once per person / waiting group"""
        matched, waiting = set(), set()
        for emb, face, img in faces:
            pid, kind, _ = self.process_face(emb, face, img)
            if kind == 'waiting':
                waiting.add(pid)
            else:
                matched.add(pid)

        for pid in matched:
            self._copy_into(src, OUTPUT_DIR / self.db.person_metadata[pid]['name'])

        waiting -= matched  # already sorted confidently into that person
        for pid in waiting:
            self.copy_to_waiting(src, pid)

        self.stats['faces_detected'] += len(faces)
        if waiting:
            self.stats['waiting_count'] += 1
        self.stats['persons_found'] = len(self.db.person_metadata)

        names = [self.db.person_metadata[p]['name'] for p in matched]
        return ('waiting' if waiting and not matched else 'sorted'), names

    def _handle_no_face(self, src: Path):
        self._copy_into(src, WAITING_DIR / "no_face_detected")

    def process_file(self, src: Path, is_video: bool):
        """Process a single file. Returns False if stopped mid-file."""
        if is_video:
            faces = self.get_faces_from_video(str(src))
            if faces is None:
                return False  # stopped: don't sort partial results, don't mark processed
        else:
            faces = self.get_faces_from_image(str(src))

        if not faces:
            self.log(f"  No face detected in {src.name}")
            self._handle_no_face(src)
            self.db.mark_processed(src, 'no_face')
            return True

        label = "unique face(s) in video" if is_video else "face(s)"
        self.log(f"  Found {len(faces)} {label}")
        status, names = self.sort_file(src, faces)
        self.db.mark_processed(src, status, names)
        return True

    def process_all(self):
        self.is_running = True
        self.should_stop = False
        self.reset_stats('running')

        try:
            if not self.initialize_model():
                self.stats['status'] = 'error'
                return

            all_files = sorted(
                f for f in CLEAN_DIR.iterdir()
                if f.is_file() and f.suffix.lower() in (SUPPORTED_IMAGES | SUPPORTED_VIDEOS)
            )

            if not all_files:
                self.log("No files found in Clean folder", 'error')
                self.stats['status'] = 'idle'
                return

            pending = [f for f in all_files if not self.db.is_processed(f)]
            skipped = len(all_files) - len(pending)

            self.stats['total_files'] = len(all_files)
            self.stats['processed_files'] = skipped
            self.stats['skipped_files'] = skipped
            self.stats['persons_found'] = len(self.db.person_metadata)

            self.log(f"Found {len(all_files)} files  |  already processed: {skipped}  |  to do: {len(pending)}")

            images = [f for f in pending if f.suffix.lower() in SUPPORTED_IMAGES]
            videos = [f for f in pending if f.suffix.lower() in SUPPORTED_VIDEOS]
            self.log(f"Images: {len(images)}  |  Videos: {len(videos)}  "
                     f"(scan first {VIDEO_SCAN_SECONDS:g}s, every {VIDEO_FRAME_INTERVAL:g}s)")

            since_save = 0
            for group_name, files, is_video in (("Images", images, False), ("Videos", videos, True)):
                if not files or self.should_stop:
                    continue
                self.log(f"--- Processing {group_name} ---")

                for f in files:
                    if self.should_stop:
                        break

                    self.stats['current_file'] = f.name
                    self.log(f"{'Video' if is_video else 'Image'}: {f.name}")

                    try:
                        if not self.process_file(f, is_video):
                            break  # stopped mid-video
                    except Exception as e:
                        self.log(f"  Error on {f.name}: {e}", 'error')
                        logger.debug(traceback.format_exc())

                    self.stats['processed_files'] += 1
                    since_save += 1
                    if since_save >= SAVE_EVERY_N_FILES:
                        self.db.save()
                        since_save = 0

                    if self.progress_callback:
                        self.progress_callback(self.stats)

            self.stats['current_file'] = ''
            if self.should_stop:
                self.stats['status'] = 'stopped'
                self.log("Processing stopped. Progress saved — press Start to resume.")
            else:
                self.stats['status'] = 'completed'
                self.log("Processing complete!")
            self.log(f"  Persons found: {len(self.db.person_metadata)}")
            self.log(f"  Faces detected: {self.stats['faces_detected']}")
            self.log(f"  Waiting review: {self.stats['waiting_count']} files")

        except Exception:
            self.log(f"Error: {traceback.format_exc()}", 'error')
            self.stats['status'] = 'error'
        finally:
            self.db.save()  # always checkpoint, even on stop / crash
            self.is_running = False
            if self.progress_callback:
                self.progress_callback(self.stats)

    def start(self):
        if self.is_running:
            return False
        self.is_running = True  # set before thread starts to avoid double start
        thread = threading.Thread(target=self.process_all, daemon=True)
        thread.start()
        return True

    def stop(self):
        self.should_stop = True
        self.log("Stopping processing...")

    def get_stats(self):
        self.stats['persons_found'] = len(self.db.person_metadata)
        return self.stats

    def get_persons(self):
        return self.db.get_all_persons()

    def reset_processed(self):
        """Forget which files were processed so the next run re-scans everything"""
        if self.is_running:
            return False
        self.db.clear_processed()
        return True

    def rename_person(self, person_id, new_name):
        try:
            if person_id not in self.db.person_metadata:
                return False

            old_name = self.db.person_metadata[person_id]['name']
            old_folder = OUTPUT_DIR / old_name
            new_folder = OUTPUT_DIR / new_name

            if new_folder.exists() and old_folder.exists() and old_folder != new_folder:
                return False  # target name already used by another folder

            if old_folder.exists():
                shutil.move(str(old_folder), str(new_folder))

            self.db.rename_person(person_id, new_name)
            return True
        except Exception as e:
            logger.error(f"Error renaming person: {e}")
            return False

    def merge_persons(self, from_id, to_id):
        try:
            if from_id == to_id:
                return False
            if from_id not in self.db.person_metadata:
                return False
            if to_id not in self.db.person_metadata:
                return False

            from_name = self.db.person_metadata[from_id]['name']
            to_name = self.db.person_metadata[to_id]['name']

            from_folder = OUTPUT_DIR / from_name
            to_folder = OUTPUT_DIR / to_name
            to_folder.mkdir(exist_ok=True)

            if from_folder.exists():
                for f in from_folder.iterdir():
                    dest = to_folder / f.name
                    if not dest.exists():
                        shutil.move(str(f), str(dest))
                shutil.rmtree(str(from_folder), ignore_errors=True)

            self.db.merge(from_id, to_id)
            return True
        except Exception as e:
            logger.error(f"Error merging persons: {e}")
            return False


engine = None


def get_engine():
    global engine
    if engine is None:
        engine = FaceRecognitionEngine()
    return engine
