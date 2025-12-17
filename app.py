import os
import json
import threading
import time
import hashlib
import argparse
import multiprocessing
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request, send_file, session, redirect, url_for

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from PIL import Image, ExifTags, ImageOps
import io
import uuid

# --- INSIGHTFACE ---
try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    print("Error: pip install insightface onnxruntime opencv-python")
    exit(1)

# --- CONFIG ---
parser = argparse.ArgumentParser()
parser.add_argument('--root-photo-dir', required=True)
parser.add_argument('--port', type=int, default=5000)
args = parser.parse_args()

ROOT_PHOTO_DIR = args.root_photo_dir
PEOPLE_DB_FILE = "people.json"
# Added Video Extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.mp4', '.mov'}
MAX_WORKERS = max(1, int(os.cpu_count() * 0.6)) 
DET_THRESHOLD = 0.60
MIN_FACE_SIZE = 40

app = Flask(__name__, static_folder='static')
app.secret_key = 'replace_this_with_a_random_secret_key'



# --- GLOBAL STATE ---
global_state = {
    "photos": {}, 
    "path_map": {},
    "people": {},
    "status": {
        "is_scanning": False, "progress": 0, "current_task": "Idle", 
        "total_files": 0, "processed_files": 0
    }
}

worker_face_app = None

def init_worker():
    global worker_face_app
    worker_face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    worker_face_app.prepare(ctx_id=0, det_size=(640, 640))

# --- UTILS ---

def calculate_md5(file_path, block_size=65536):
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(block_size), b""): hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except: return None

def get_file_dates(path):
    try:
        stat = os.stat(path)
        c = datetime.datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        m = datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        return c, m
    except: return "-","-"

def extract_gps(img):
    try:
        exif = img.getexif()
        if not exif: return None
        gps_info = exif.get_ifd(0x8825)
        if not gps_info: return None
        def to_decimal(dms, ref):
            d = dms[0] + (dms[1]/60.0) + (dms[2]/3600.0)
            return -d if ref in ['S', 'W'] else d
        if 2 in gps_info and 4 in gps_info:
            lat = to_decimal(gps_info[2], gps_info.get(1, 'N'))
            lon = to_decimal(gps_info[4], gps_info.get(3, 'E'))
            return f"{lat},{lon}"
    except: pass
    return None

# --- WORKER ---

def process_single_image(file_data):
    full_path, root_dir = file_data
    filename = os.path.basename(full_path)
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        md5_val = calculate_md5(full_path)
        if not md5_val: return None
        c_time, m_time = get_file_dates(full_path)
        
        img = None
        is_video = False

        if ext in ['.mp4', '.mov']:
            is_video = True
            # Extract frame from middle of video using OpenCV
            cap = cv2.VideoCapture(full_path)
            if cap.isOpened():
                # Jump to 20% to avoid black frames at start
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(length * 0.2))
                ret, frame = cap.read()
                if ret:
                    img = frame # BGR format
            cap.release()
        else:
            # Image Processing
            try:
                pil_img = Image.open(full_path)
                pil_img = ImageOps.exif_transpose(pil_img)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except:
                img = cv2.imread(full_path)

        if img is None: return None

        faces = worker_face_app.get(img)
        faces_data = []
        for face in faces:
            if face.det_score < DET_THRESHOLD: continue
            w = face.bbox[2] - face.bbox[0]
            h = face.bbox[3] - face.bbox[1]
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE: continue

            bbox = face.bbox.astype(int).tolist()
            # [top, right, bottom, left]
            converted_box = [bbox[1], bbox[2], bbox[3], bbox[0]]
            face_id = f"{bbox[1]}_{bbox[2]}_{bbox[3]}_{bbox[0]}"

            faces_data.append({
                "id": face_id,
                "box": converted_box,
                "encoding": face.embedding.tolist(),
                "cluster_id": "-1",
                "manual": False
            })

        return {
            "status": "success", "full_path": full_path, "root_dir": root_dir, 
            "filename": filename, "md5": md5_val, "created": c_time, "modified": m_time, 
            "faces": faces_data, "is_video": is_video
        }
    except Exception as e:
        return {"status": "error", "file": filename, "msg": str(e)}

# --- SCANNER (Logic Unchanged, just passes is_video) ---

class ScannerThread(threading.Thread):
    def run(self):
        global_state["status"].update({"is_scanning": True, "progress": 0, "current_task": "Discovery"})
        files_to_process = []
        updates_by_dir = {} 
        all_files_count = 0
        
        for root, dirs, files in os.walk(ROOT_PHOTO_DIR):
            profile_path = os.path.join(root, "profile.json")
            local_cache = {}
            if os.path.exists(profile_path):
                try: 
                    with open(profile_path, 'r') as f: 
                        local_cache = json.load(f)
                except: pass
            updates_by_dir[root] = local_cache
            
            for file in files:
                if os.path.splitext(file)[1].lower() in ALLOWED_EXTENSIONS:
                    all_files_count += 1
                    full_path = os.path.join(root, file)
                    should_process = True
                    if file in local_cache:
                        c = local_cache[file]
                        if c.get("faces") and "id" not in c["faces"][0]: should_process = True
                        else:
                            should_process = False
                            self.ingest_data(full_path, c["md5"], c["faces"], c.get("created"), c.get("modified"), c.get("is_video", False))
                    if should_process: files_to_process.append((full_path, root))

        global_state["status"]["total_files"] = all_files_count
        
        if files_to_process:
            with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker) as executor:
                futures = [executor.submit(process_single_image, i) for i in files_to_process]
                done = 0
                for f in as_completed(futures):
                    res = f.result()
                    done += 1
                    global_state["status"]["progress"] = int((done / len(files_to_process))*80)
                    global_state["status"]["current_task"] = f"Processed {done}/{len(files_to_process)}"
                    if res and res["status"] == "success":
                        self.ingest_data(res["full_path"], res["md5"], res["faces"], res["created"], res["modified"], res.get("is_video", False))
                        updates_by_dir[res["root_dir"]][res["filename"]] = {
                            "md5": res["md5"], "created": res["created"], "modified": res["modified"], 
                            "faces": res["faces"], "is_video": res.get("is_video", False)
                        }

        for root, data in updates_by_dir.items():
            if data:
                with open(os.path.join(root, "profile.json"), 'w') as f: json.dump(data, f, indent=4)

        self.perform_clustering(updates_by_dir)
        global_state["status"].update({"is_scanning": False, "progress": 100, "current_task": "Done"})

    def ingest_data(self, full_path, md5, faces, c_time, m_time, is_video):
        global_state["path_map"][full_path] = md5
        folder = os.path.basename(os.path.dirname(full_path))
        parts = folder.split('_')
        location = "_".join(parts[1:]) if len(parts) > 1 else folder

        for f in faces:
            if "id" not in f and "box" in f:
                 b = f["box"]
                 f["id"] = f"{b[0]}_{b[1]}_{b[2]}_{b[3]}"

        gps_coords = None
        # Try to read GPS if we don't have it (simplified for this demo)
        # In production, save this to profile.json to avoid re-reading images
        if not is_video:
            try:
                img = Image.open(full_path)
                gps_str = extract_gps(img)
                if gps_str:
                    lat, lon = gps_str.split(',')
                    gps_coords = (float(lat), float(lon))
            except: pass

        if md5 not in global_state["photos"]:
            global_state["photos"][md5] = {
                "paths": [], "faces": faces, "location": location,
                "folder": folder, "created": c_time, "modified": m_time, 
                "filename": os.path.basename(full_path),
                "is_video": is_video,
                "gps_coords": gps_coords
            }
        if full_path not in global_state["photos"][md5]["paths"]:
            global_state["photos"][md5]["paths"].append(full_path)

    def perform_clustering(self, updates_by_dir):
        global_state["status"]["current_task"] = "Organizing..."
        
        # --- PHASE 1: BUILD ANCHORS (The "Memory") ---
        # We calculate the average face (centroid) for every person currently in the DB.
        # This includes DELETED people, so we can recognize and ignore them later.
        
        person_vectors = {} # { "person_uuid": [vector1, vector2...] }
        
        # 1a. Gather vectors from currently assigned faces
        for md5, pdata in global_state["photos"].items():
            for face in pdata["faces"]:
                cid = face["cluster_id"]
                # Only use valid encodings assigned to a known person ID
                if cid != "-1" and cid in global_state["people"]:
                    if len(face["encoding"]) == 512:
                        if cid not in person_vectors: person_vectors[cid] = []
                        person_vectors[cid].append(face["encoding"])

        # 1b. Calculate Centroids (The "Average Face")
        anchors = {} # { "person_uuid": numpy_array_centroid }
        for cid, vectors in person_vectors.items():
            if len(vectors) > 0:
                anchors[cid] = np.mean(vectors, axis=0)

        # --- PHASE 2: SEPARATE UNKNOWNS ---
        unknown_faces = [] # List of { "md5":..., "idx":..., "encoding":... }
        
        for md5, pdata in global_state["photos"].items():
            for i, face in enumerate(pdata["faces"]):
                # We process faces that are Unknown (-1) AND not manually "un-tagged"
                # If user manually set it to -1 (manual=True), we respect that and skip it.
                if face["cluster_id"] == "-1" and not face.get("manual", False) and not face.get("is_deleted", False):
                    if len(face["encoding"]) == 512:
                        unknown_faces.append({
                            "md5": md5,
                            "idx": i,
                            "encoding": face["encoding"]
                        })

        if not unknown_faces: return

        print(f"Processing {len(unknown_faces)} new faces against {len(anchors)} existing people...")
        
        # Prepare data for bulk updates
        dirty_dirs = set()
        faces_to_cluster = [] # Faces that didn't match anyone (for DBSCAN)

        # --- PHASE 3: RECOGNITION (Match against Anchors) ---
        if anchors:
            anchor_ids = list(anchors.keys())
            anchor_matrix = np.array(list(anchors.values()))
            unknown_matrix = np.array([f["encoding"] for f in unknown_faces])
            
            # Calculate distance matrix (Unknowns vs Anchors)
            # Result is [num_unknowns x num_anchors]
            dists = cosine_distances(unknown_matrix, anchor_matrix)
            
            # Threshold for "Same Person" (0.4 is standard for ArcFace)
            MATCH_THRESHOLD = 0.40 
            
            for i, face_data in enumerate(unknown_faces):
                # Find closest anchor
                min_dist_idx = np.argmin(dists[i])
                min_dist = dists[i][min_dist_idx]
                
                if min_dist < MATCH_THRESHOLD:
                    # MATCH FOUND!
                    matched_pid = anchor_ids[min_dist_idx]
                    matched_person = global_state["people"][matched_pid]
                    
                    md5 = face_data["md5"]
                    f_idx = face_data["idx"]
                    
                    # Check if this person was deleted
                    if matched_person.get("is_deleted", False):
                        # Auto-delete this face
                        global_state["photos"][md5]["faces"][f_idx]["is_deleted"] = True
                        global_state["photos"][md5]["faces"][f_idx]["manual"] = True # Lock it
                        global_state["photos"][md5]["faces"][f_idx]["cluster_id"] = "-1"
                    else:
                        # Assign to existing person
                        global_state["photos"][md5]["faces"][f_idx]["cluster_id"] = matched_pid
                    
                    # Update Disk Buffer
                    self._mark_dirty(md5, f_idx, updates_by_dir, dirty_dirs)
                else:
                    # No match found, send to clustering pool
                    faces_to_cluster.append(face_data)
        else:
            # No existing people, everyone goes to clustering
            faces_to_cluster = unknown_faces

        # --- PHASE 4: DISCOVERY (DBSCAN on leftovers) ---
        if len(faces_to_cluster) > 0:
            print(f"Clustering {len(faces_to_cluster)} remaining unknown faces...")
            cluster_matrix = [f["encoding"] for f in faces_to_cluster]
            
            # Run DBSCAN
            clt = DBSCAN(metric="cosine", n_jobs=-1, eps=0.40, min_samples=2)
            clt.fit(cluster_matrix)
            
            # Assign new UUIDs to new clusters
            new_cluster_map = {} # { dbscan_label: new_uuid }
            
            for i, label_id in enumerate(clt.labels_):
                if label_id == -1: continue # Noise, stays unknown
                
                face_data = faces_to_cluster[i]
                md5 = face_data["md5"]
                f_idx = face_data["idx"]
                
                # Create new Person ID if we haven't seen this label yet
                if label_id not in new_cluster_map:
                    new_uid = uuid.uuid4().hex
                    new_cluster_map[label_id] = new_uid
                    
                    # Create Entry in People DB
                    first_face = global_state["photos"][md5]["faces"][f_idx]
                    global_state["people"][new_uid] = {
                        "name": f"New Person {str(new_uid)[:6]}", 
                        "gender": "Unknown",
                        "avatar_md5": md5, 
                        "avatar_face_id": first_face.get("id"),
                        "is_deleted": False
                    }
                    save_people_db()
                
                # Assign
                final_pid = new_cluster_map[label_id]
                global_state["photos"][md5]["faces"][f_idx]["cluster_id"] = final_pid
                
                # Update Disk Buffer
                self._mark_dirty(md5, f_idx, updates_by_dir, dirty_dirs)

        # --- PHASE 5: SAVE TO DISK ---
        for d in dirty_dirs:
            p_path = os.path.join(d, "profile.json")
            if d in updates_by_dir:
                with open(p_path, 'w') as f: json.dump(updates_by_dir[d], f, indent=4)

    def _mark_dirty(self, md5, f_idx, updates_by_dir, dirty_dirs):
        """Helper to update the disk write buffer"""
        if global_state["photos"][md5]["paths"]:
            main_path = global_state["photos"][md5]["paths"][0]
            d = os.path.dirname(main_path)
            f = os.path.basename(main_path)
            
            if d in updates_by_dir and f in updates_by_dir[d]:
                # Sync memory state to disk buffer
                mem_face = global_state["photos"][md5]["faces"][f_idx]
                disk_face_list = updates_by_dir[d][f]["faces"]
                
                if f_idx < len(disk_face_list):
                    disk_face_list[f_idx]["cluster_id"] = mem_face["cluster_id"]
                    disk_face_list[f_idx]["manual"] = mem_face.get("manual", False)
                    disk_face_list[f_idx]["is_deleted"] = mem_face.get("is_deleted", False)
                    dirty_dirs.add(d)


SETTINGS_FILE = "settings.json"

def check_credentials(username, password):
    # Create default if missing
    if not os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'w') as f:
            json.dump({"users": {"admin": "pass"}}, f, indent=4)
    
    try:
        with open(SETTINGS_FILE, 'r') as f:
            config = json.load(f)
            users = config.get("users", {})
            return users.get(username) == password
    except:
        return False

@app.before_request
def require_login():
    # 1. Allowed endpoints (Login, Static files, Logout)
    if request.endpoint in ['login', 'static', 'logout']:
        return
    
    # 2. If user is logged in, proceed
    if 'user' in session:
        return

    # 3. Otherwise, redirect to login
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('username')
        pw = request.form.get('password')
        if check_credentials(user, pw):
            session['user'] = user
            return redirect(url_for('index'))
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# --- ROUTES ---

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/refresh_library')
def refresh():
    if not global_state["status"]["is_scanning"]:
        ScannerThread().start()
        return jsonify({"status": "started"})
    return jsonify({"status": "busy"})

@app.route('/api/scan_status')
def status(): return jsonify(global_state["status"])

@app.route('/api/directories')
def get_dirs():
    res = {}
    for md5, p in global_state["photos"].items():
        f = p["folder"]
        if f not in res: 
            # Use this photo as the cover for the folder
            # Prefer video thumb if it's a video, else image
            cover_url = f"/video_thumb/{md5}" if p.get("is_video") else f"/image_by_md5/{md5}"
            res[f] = {
                "name": f, 
                "location": p["location"], 
                "count": 0, 
                "cover": cover_url
            }
        res[f]["count"] += 1
    return jsonify(sorted(list(res.values()), key=lambda x:x['name'], reverse=True))

@app.route('/api/photos_by_dir')
def photos_by_dir():
    folder = request.args.get('folder')
    res = []
    for md5, p in global_state["photos"].items():
        if p["folder"] == folder:
            faces_simple = []
            for face in p["faces"]:
                cid = face["cluster_id"]
                name = global_state["people"][cid]["name"] if cid in global_state["people"] else "Unknown"
                faces_simple.append({"name": name, "face_id": face["id"], "md5": md5})
            res.append({
                "md5": md5, "url": f"/video_thumb/{md5}" if p.get("is_video") else f"/image_by_md5/{md5}",
                "filename": p["filename"], "faces": faces_simple, "is_video": p.get("is_video", False)
            })
    return jsonify(res)

@app.route('/api/people')
def get_people():
    res = []
    counts = {}
    for md5, p in global_state["photos"].items():
        for f in p["faces"]:
            if not f.get("is_deleted", False):
                counts[f["cluster_id"]] = counts.get(f["cluster_id"], 0) + 1
    for cid, p in global_state["people"].items():
        if cid == "-1" or p.get("is_deleted", False): continue
        if counts.get(cid, 0) == 0: continue
        res.append({
            "id": cid, "name": p["name"], "gender": p["gender"],
            "count": counts.get(cid, 0),
            "avatar_url": f"/api/face_crop/{p['avatar_md5']}/{p['avatar_face_id']}"
})
    return jsonify(sorted(res, key=lambda x:x['count'], reverse=True))

@app.route('/api/faces_by_person/<pid>')
def faces_by_person(pid):
    res = []
    for md5, p in global_state["photos"].items():
        for f in p["faces"]:
            if f["cluster_id"] == pid:
                res.append({
                    "md5": md5, "face_id": f["id"],
                    "url": f"/api/face_crop/{md5}/{f['id']}",
                    "parent_url": f"/image_by_md5/{md5}"
                })
    return jsonify(res)

@app.route('/api/set_avatar', methods=['POST'])
def set_avatar():
    d = request.json
    person_id = d.get('person_id')
    md5 = d.get('md5')
    face_id = d.get('face_id')
    
    if person_id in global_state["people"]:
        global_state["people"][person_id]["avatar_md5"] = md5
        global_state["people"][person_id]["avatar_face_id"] = face_id
        save_people_db()
        return jsonify({"success": True})
    return jsonify({"success": False})

@app.route('/api/map_all')
def map_all():
    points = []
    for md5, p in global_state["photos"].items():
        if not p["paths"]: continue
        
        # Ensure we have GPS
        if p.get("gps_coords"):
            lat, lon = p["gps_coords"]
            thumb = f"/video_thumb/{md5}" if p.get("is_video") else f"/image_by_md5/{md5}"
            points.append({
                "lat": lat, 
                "lon": lon, 
                "md5": md5, 
                "thumb": thumb,
                "created": p.get("created", "0") # <--- Added for sorting
            })
            
    # Optional: Sort by date descending globally (helpful, but clustering logic does local sort)
    points.sort(key=lambda x: x['created'], reverse=True)
    return jsonify(points)

# --- IMAGE/VIDEO SERVING ---

@app.route('/image_by_md5/<md5>')
def serve_img(md5):
    if md5 in global_state["photos"] and global_state["photos"][md5]["paths"]:
        return send_file(global_state["photos"][md5]["paths"][0])
    return "Missing", 404

@app.route('/video_thumb/<md5>')
def serve_video_thumb(md5):
    """
    Dynamically extracts a frame from the video and serves it as JPEG.
    No disk save.
    """
    if md5 not in global_state["photos"]: return "Err", 404
    path = global_state["photos"][md5]["paths"][0]
    
    try:
        cap = cv2.VideoCapture(path)
        # Seek to 10% to capture a representative frame
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.1))
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert BGR (OpenCV) to RGB (Pillow)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            
            # Create in-memory stream
            bio = io.BytesIO()
            img.save(bio, 'JPEG', quality=70)
            bio.seek(0)
            return send_file(bio, mimetype='image/jpeg')
    except: pass
    
    # Fallback/Error placeholder
    return send_file('static/img/folder.svg') 


@app.route('/api/face_crop/<md5>/<face_id>')
def serve_crop(md5, face_id):
    if md5 not in global_state["photos"]: return "Err", 404
    
    # 1. Locate Face
    target_face = None
    for f in global_state["photos"][md5]["faces"]:
        if f["id"] == face_id:
            target_face = f
            break
            
    if not target_face: return "Face Not Found", 404
    
    path = global_state["photos"][md5]["paths"][0]
    p_data = global_state["photos"][md5]
    img = None
    
    try:
        # 2. Load Image/Video
        if p_data.get("is_video"):
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.2))
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
        else:
            img = Image.open(path).convert("RGB")
            img = ImageOps.exif_transpose(img)
        
        if img:
            # 3. Calculate Perfect Square Crop
            # stored box is [top, right, bottom, left]
            top, right, bottom, left = target_face["box"]
            
            face_w = right - left
            face_h = bottom - top
            center_x = left + (face_w / 2)
            center_y = top + (face_h / 2)
            
            # Make a square box: 1.5x the largest dimension of the face
            box_size = max(face_w, face_h) * 1.5
            half_box = box_size / 2
            
            # Calculate coordinates (might be negative or larger than image)
            x1 = int(center_x - half_box)
            y1 = int(center_y - half_box)
            x2 = int(center_x + half_box)
            y2 = int(center_y + half_box)
            
            # 4. Safe Crop with Padding
            # If coordinates are OOB, we pad the image instead of clipping the box
            # This ensures the face stays in the center of the result
            img_w, img_h = img.size
            
            # Check if padding is needed
            pad_l = abs(min(0, x1))
            pad_t = abs(min(0, y1))
            pad_r = max(0, x2 - img_w)
            pad_b = max(0, y2 - img_h)
            
            if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
                # Add padding (black background)
                new_w = img_w + pad_l + pad_r
                new_h = img_h + pad_t + pad_b
                new_img = Image.new("RGB", (new_w, new_h), (0, 0, 0))
                new_img.paste(img, (pad_l, pad_t))
                
                # Adjust crop coordinates to new padded space
                x1 += pad_l
                y1 += pad_t
                x2 += pad_l
                y2 += pad_t
                
                crop = new_img.crop((x1, y1, x2, y2))
            else:
                # Standard crop
                crop = img.crop((x1, y1, x2, y2))

            # 5. Resize for thumbnail performance
            crop.thumbnail((250, 250), Image.Resampling.LANCZOS)
            
            bio = io.BytesIO()
            crop.save(bio, 'JPEG', quality=90)
            bio.seek(0)
            return send_file(bio, mimetype='image/jpeg')
            
    except Exception as e:
        print(f"Crop Error: {e}")
        pass # Fallback
        
    return send_file('static/img/people.svg')

@app.route('/api/photo_details/<md5>')
def details(md5):
    if md5 not in global_state["photos"]: return jsonify({}), 404
    p = global_state["photos"][md5]
    
    exif = {}
    gps = None
    try:
        if not p.get("is_video"):
            img = Image.open(p["paths"][0])
            gps = extract_gps(img)
            raw = img.getexif()
            if raw:
                for k,v in raw.items():
                    tag = ExifTags.TAGS.get(k, k)
                    if len(str(v)) < 100: exif[tag] = str(v)
    except: pass
    
    faces = []
    for f in p["faces"]:
        cid = f["cluster_id"]
        # LOOKUP NAME
        name = "Unknown"
        if cid in global_state["people"]:
            name = global_state["people"][cid]["name"]
            if global_state["people"][cid].get("is_deleted"):
                name += " (Deleted)"

        faces.append({
            "name": name,
            "face_id": f["id"], 
            "md5": md5,
            "cluster_id": cid  # <--- ADDED THIS
        })

    return jsonify({
        "filename": p["filename"], "folder": p["folder"], "location": p["location"],
        "gps": gps, "created": p["created"], "exif": exif, "faces": faces, 
        "full_path_id": md5, "is_video": p.get("is_video", False)
    })

# --- DATA MANIPULATION (Keep existing remove/delete/update/merge routes) ---
@app.route('/api/remove_face_from_cluster', methods=['POST'])
def remove_face():
    d = request.json
    md5 = d.get('md5')
    fid = d.get('face_id')
    if md5 in global_state["photos"]:
        for f in global_state["photos"][md5]["faces"]:
            if f["id"] == fid:
                f["cluster_id"] = "-1"
                f["manual"] = True
                save_profile_for_md5(md5)
                return jsonify({"success": True})
    return jsonify({"success": False})

@app.route('/api/delete_people_bulk', methods=['POST'])
def delete_bulk():
    ids = set(request.json.get('ids', []))
    for md5, p in global_state["photos"].items():
        changed = False
        for f in p["faces"]:
            if f["cluster_id"] in ids:
                f["is_deleted"] = True; f["manual"] = True; f["cluster_id"] = "-1"; changed = True
        if changed: save_profile_for_md5(md5)
    for i in ids: 
        if i in global_state["people"]: global_state["people"][i]["is_deleted"] = True
    save_people_db()
    return jsonify({"success": True})

@app.route('/api/merge_people', methods=['POST'])
def merge_people():
    d = request.json
    sid, tid = d.get('source_id'), d.get('target_id')
    if sid not in global_state["people"] or tid not in global_state["people"]: return jsonify({"success": False})
    for md5, p in global_state["photos"].items():
        changed = False
        for f in p["faces"]:
            if f["cluster_id"] == sid:
                f["cluster_id"] = tid; f["manual"] = True; changed = True
        if changed: save_profile_for_md5(md5)
    if sid in global_state["people"]: del global_state["people"][sid]
    save_people_db()
    return jsonify({"success": True})

# --- UTILS FOR SAVING ---
def save_profile_for_md5(md5):
    if md5 in global_state["photos"]:
        path = global_state["photos"][md5]["paths"][0]
        d_path = os.path.dirname(path)
        p_path = os.path.join(d_path, "profile.json")
        try:
            with open(p_path, 'r') as f: data = json.load(f)
            fname = os.path.basename(path)
            if fname in data:
                mem_faces = global_state["photos"][md5]["faces"]
                for mf in mem_faces:
                    for df in data[fname]["faces"]:
                        if "id" in df and df["id"] == mf["id"]:
                            df["cluster_id"] = mf["cluster_id"]; df["manual"] = mf["manual"]; df.get("is_deleted", False)
            with open(p_path, 'w') as f: json.dump(data, f, indent=4)
        except: pass

def load_people_db():
    if os.path.exists(PEOPLE_DB_FILE):
        with open(PEOPLE_DB_FILE) as f: global_state["people"] = json.load(f)

def save_people_db():
    with open(PEOPLE_DB_FILE, 'w') as f: json.dump(global_state["people"], f, indent=4)

@app.route('/api/search')
def search():
    query = request.args.get('q', '').lower().strip()
    if not query: return jsonify([])
    results = []
    people_map = {cid: p['name'].lower() for cid, p in global_state['people'].items() if not p.get('is_deleted')}
    for md5, p in global_state["photos"].items():
        match = False
        if (query in p['filename'].lower() or query in p['location'].lower() or query in p['folder'].lower() or query in p['created']): match = True
        if not match:
            for face in p['faces']:
                cid = face['cluster_id']
                if cid in people_map and query in people_map[cid]: match = True; break
        if match:
            faces_simple = []
            for face in p["faces"]:
                cid = face["cluster_id"]
                name = global_state["people"][cid]["name"] if cid in global_state["people"] else "Unknown"
                faces_simple.append({"name": name, "face_id": face["id"], "md5": md5})
            results.append({
                "md5": md5, "url": f"/video_thumb/{md5}" if p.get("is_video") else f"/image_by_md5/{md5}",
                "filename": p["filename"], "created": p["created"], "faces": faces_simple, "is_video": p.get("is_video", False)
            })
    results.sort(key=lambda x: x['created'], reverse=True)
    return jsonify(results)


@app.route('/api/person_details/<pid>')
def get_person_details(pid):
    if pid in global_state["people"]:
        p = global_state["people"][pid]
        return jsonify({
            "id": pid,
            "name": p["name"],
            "gender": p["gender"],
            "is_deleted": p.get("is_deleted", False),
            "avatar_url": f"/api/face_crop/{p['avatar_md5']}/{p['avatar_face_id']}"
        })
    return jsonify({"error": "Not found"}), 404

@app.route('/api/update_person', methods=['POST'])
def update_person():
    d = request.json
    cid = d.get('id')
    if cid in global_state["people"]:
        global_state["people"][cid].update({
            "name": d.get('name'), 
            "gender": d.get('gender'),
            "is_deleted": False  # <--- FORCE RESTORE ON EDIT
        })
        save_people_db()
        return jsonify({"success": True})
    return jsonify({"success": False})

# Add uuid import if missing
import uuid

@app.route('/api/create_person', methods=['POST'])
def create_person():
    d = request.json
    md5 = d.get('md5')
    face_id = d.get('face_id')
    name = d.get('name')
    gender = d.get('gender')

    # 1. Create new Person ID
    new_pid = uuid.uuid4().hex
    
    global_state["people"][new_pid] = {
        "name": name,
        "gender": gender,
        "avatar_md5": md5,
        "avatar_face_id": face_id,
        "is_deleted": False
    }
    save_people_db()

    # 2. Update the specific face in memory
    if md5 in global_state["photos"]:
        for f in global_state["photos"][md5]["faces"]:
            if f["id"] == face_id:
                f["cluster_id"] = new_pid
                f["manual"] = True # Lock it so AI doesn't change it back
                break
        # 3. Save to profile.json
        save_profile_for_md5(md5)

    return jsonify({"success": True, "new_id": new_pid})

@app.route('/api/assign_face', methods=['POST'])
def assign_face():
    d = request.json
    md5 = d.get('md5')
    face_id = d.get('face_id')
    target_id = d.get('target_id') # The existing person UUID

    if target_id not in global_state["people"]:
        return jsonify({"success": False, "error": "Person not found"})

    if md5 in global_state["photos"]:
        for f in global_state["photos"][md5]["faces"]:
            if f["id"] == face_id:
                f["cluster_id"] = target_id
                f["manual"] = True
                f["is_deleted"] = False # Restore if it was deleted
                break
        save_profile_for_md5(md5)

    return jsonify({"success": True})

if __name__ == '__main__':
    multiprocessing.freeze_support()
    load_people_db()
    app.run(debug=True, port=args.port)