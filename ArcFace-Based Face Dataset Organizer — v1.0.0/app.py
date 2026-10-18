"""
Flask API Server for ArcFace Dataset Sorter
Serves Web GUI and handles all API requests
"""

import os
import json
import base64
from pathlib import Path
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from face_recognizer import get_engine, OUTPUT_DIR, WAITING_DIR, THUMBNAILS_DIR, BASE_DIR

app = Flask(__name__)
CORS(app)

# ─── Static / GUI ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


# ─── API: Processing Control ──────────────────────────────────────────────────

@app.route('/api/start', methods=['POST'])
def start_processing():
    eng = get_engine()
    if eng.is_running:
        return jsonify({'success': False, 'message': 'Already running'})
    
    # Reset stats
    eng.reset_stats('running')
    
    success = eng.start()
    return jsonify({'success': success})


@app.route('/api/stop', methods=['POST'])
def stop_processing():
    eng = get_engine()
    eng.stop()
    return jsonify({'success': True})


@app.route('/api/reset-processed', methods=['POST'])
def reset_processed():
    """Forget processed-file history so the next Start re-scans every file"""
    eng = get_engine()
    success = eng.reset_processed()
    return jsonify({'success': success,
                    'message': '' if success else 'Cannot reset while running'})


@app.route('/api/status', methods=['GET'])
def get_status():
    eng = get_engine()
    stats = eng.get_stats()
    
    # Calculate progress percentage
    total = stats.get('total_files', 0)
    processed = stats.get('processed_files', 0)
    progress = int((processed / total * 100)) if total > 0 else 0
    
    return jsonify({
        **stats,
        'progress': progress
    })


# ─── API: Persons ─────────────────────────────────────────────────────────────

@app.route('/api/persons', methods=['GET'])
def get_persons():
    eng = get_engine()
    persons = eng.get_persons()
    return jsonify({'persons': persons})


@app.route('/api/persons/<int:person_id>/rename', methods=['POST'])
def rename_person(person_id):
    eng = get_engine()
    data = request.json
    new_name = data.get('name', '').strip()
    
    if not new_name:
        return jsonify({'success': False, 'message': 'Name cannot be empty'})
    
    # Sanitize name
    import re
    new_name = re.sub(r'[<>:"/\\|?*]', '_', new_name)
    
    success = eng.rename_person(person_id, new_name)
    return jsonify({'success': success})


@app.route('/api/persons/merge', methods=['POST'])
def merge_persons():
    eng = get_engine()
    data = request.json
    from_id = data.get('from_id')
    to_id = data.get('to_id')
    
    if not from_id or not to_id:
        return jsonify({'success': False, 'message': 'Missing person IDs'})
    
    success = eng.merge_persons(int(from_id), int(to_id))
    return jsonify({'success': success})


# ─── API: Thumbnails ──────────────────────────────────────────────────────────

@app.route('/api/thumbnail/<int:person_id>', methods=['GET'])
def get_thumbnail(person_id):
    thumb_path = THUMBNAILS_DIR / f"person_{person_id:04d}.jpg"
    if thumb_path.exists():
        return send_file(str(thumb_path), mimetype='image/jpeg')
    
    # Return a placeholder if not found
    return jsonify({'error': 'Thumbnail not found'}), 404


# ─── API: Folders / Files ─────────────────────────────────────────────────────

@app.route('/api/folders', methods=['GET'])
def get_folders():
    result = []
    
    # Sorted persons
    if OUTPUT_DIR.exists():
        for folder in sorted(OUTPUT_DIR.iterdir()):
            if folder.is_dir():
                files = list(folder.glob('*'))
                result.append({
                    'type': 'sorted',
                    'name': folder.name,
                    'path': str(folder),
                    'count': len(files)
                })
    
    # Waiting folder
    if WAITING_DIR.exists():
        for folder in sorted(WAITING_DIR.iterdir()):
            if folder.is_dir():
                files = list(folder.glob('*'))
                result.append({
                    'type': 'waiting',
                    'name': folder.name,
                    'path': str(folder),
                    'count': len(files)
                })
    
    return jsonify({'folders': result})



@app.route('/api/open-folder', methods=['POST'])
def open_folder():
    data = request.json
    folder_name = data.get('path', '')
    
    # Try as absolute path first, then relative to Sorted dir
    abs_path = Path(folder_name)
    if not abs_path.is_absolute():
        abs_path = OUTPUT_DIR / folder_name.replace('Sorted\\', '').replace('Sorted/', '')
    
    if abs_path.exists():
        import subprocess
        subprocess.Popen(['explorer', str(abs_path)])
        return jsonify({'success': True})
    
    return jsonify({'success': False, 'message': f'Path not found: {abs_path}'})


@app.route('/api/open-output', methods=['GET'])
def open_output():
    import subprocess
    OUTPUT_DIR.mkdir(exist_ok=True)
    subprocess.Popen(['explorer', str(OUTPUT_DIR)])
    return jsonify({'success': True})


@app.route('/api/open-waiting', methods=['GET'])
def open_waiting():
    import subprocess
    WAITING_DIR.mkdir(exist_ok=True)
    subprocess.Popen(['explorer', str(WAITING_DIR)])
    return jsonify({'success': True})


@app.route('/api/settings', methods=['GET'])
def get_settings():
    import face_recognizer as fr  # read live values (may have been changed at runtime)
    return jsonify({
        'similarity_threshold': fr.SIMILARITY_THRESHOLD,
        'waiting_threshold': fr.WAITING_THRESHOLD,
        'video_frame_interval': fr.VIDEO_FRAME_INTERVAL,
        'video_scan_seconds': fr.VIDEO_SCAN_SECONDS,
        'min_confidence': fr.MIN_CONFIDENCE,
        'min_face_size': fr.MIN_FACE_SIZE
    })


@app.route('/api/settings', methods=['POST'])
def update_settings():
    import face_recognizer as fr
    data = request.json
    
    if 'similarity_threshold' in data:
        fr.SIMILARITY_THRESHOLD = float(data['similarity_threshold'])
    if 'waiting_threshold' in data:
        fr.WAITING_THRESHOLD = float(data['waiting_threshold'])
    if 'video_frame_interval' in data:
        fr.VIDEO_FRAME_INTERVAL = float(data['video_frame_interval'])
    if 'video_scan_seconds' in data:
        fr.VIDEO_SCAN_SECONDS = float(data['video_scan_seconds'])
    if 'min_confidence' in data:
        fr.MIN_CONFIDENCE = float(data['min_confidence'])
    
    return jsonify({'success': True})


@app.route('/api/stats/summary', methods=['GET'])
def get_summary():
    persons_count = 0
    files_sorted = 0
    
    if OUTPUT_DIR.exists():
        for folder in OUTPUT_DIR.iterdir():
            if folder.is_dir():
                persons_count += 1
                files_sorted += len(list(folder.glob('*')))
    
    waiting_count = 0
    if WAITING_DIR.exists():
        for folder in WAITING_DIR.iterdir():
            if folder.is_dir():
                waiting_count += len(list(folder.glob('*')))
    
    clean_count = 0
    clean_dir = BASE_DIR / "Clean"
    if clean_dir.exists():
        clean_count = len([f for f in clean_dir.iterdir() if f.is_file()])
    
    return jsonify({
        'persons_count': persons_count,
        'files_sorted': files_sorted,
        'waiting_count': waiting_count,
        'clean_count': clean_count
    })


if __name__ == '__main__':
    print("=" * 60)
    print("  ArcFace Dataset Sorter Server")
    print("  Open: http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
