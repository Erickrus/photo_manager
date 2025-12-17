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
from flask import Flask, render_template, jsonify, request, send_file
from sklearn.cluster import DBSCAN
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
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
MAX_WORKERS = max(1, int(os.cpu_count() * 0.6)) 
DET_THRESHOLD = 0.60
MIN_FACE_SIZE = 40

app = Flask(__name__, static_folder='static')

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

# def extract_gps(img):
#     try:
#         exif = img.getexif()
#         if not exif: return None
#         gps_info = exif.get_ifd(0x8825)
#         if not gps_info: return None
#         def to_deg(dms, ref):
#             d = dms[0] + (dms[1]/60.0) + (dms[2]/3600.0)
#             return -d if ref in ['S','W'] else d
#         if 2 in gps_info and 4 in gps_info:
#             lat = to_deg(gps_info[2], gps_info.get(1,'N'))
#             lon = to_deg(gps_info[4], gps_info.get(3,'E'))
#             return f"{lat},{lon}"
#     except: pass
#     return None



def extract_gps(img):
    """
    Extracts GPS location from PIL image and converts to decimal format.
    Returns string "lat,lon" or None.
    """
    try:
        exif = img.getexif()
        if not exif: return None
        
        # GPS Information is stored in a specific IFD (ID: 34853 / 0x8825)
        gps_info = exif.get_ifd(0x8825)
        if not gps_info: return None

        # Helper to convert DMS tuple to decimal
        def to_decimal(dms, ref):
            degrees = dms[0]
            minutes = dms[1]
            seconds = dms[2]
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            if ref in ['S', 'W']:
                decimal = -decimal
            return decimal

        # Tags: 1=LatRef, 2=Lat, 3=LonRef, 4=Lon
        if 2 in gps_info and 4 in gps_info:
            lat = to_decimal(gps_info[2], gps_info.get(1, 'N'))
            lon = to_decimal(gps_info[4], gps_info.get(3, 'E'))
            return f"{lat},{lon}"
            
    except Exception as e:
        print(f"GPS Error: {e}")
    return None
# --- WORKER ---

def process_single_image(file_data):
    full_path, root_dir = file_data
    filename = os.path.basename(full_path)
    try:
        md5_val = calculate_md5(full_path)
        if not md5_val: return None
        c_time, m_time = get_file_dates(full_path)
        
        # EXIF Rotation
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
            
            # STABLE FACE ID: based on coordinates
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
            "faces": faces_data
        }
    except Exception as e:
        return {"status": "error", "file": filename, "msg": str(e)}

# --- SCANNER ---

class ScannerThread(threading.Thread):
    def run(self):
        global_state["status"].update({"is_scanning": True, "progress": 0, "current_task": "Discovery"})
        
        files_to_process = []
        updates_by_dir = {} 
        all_files_count = 0
        
        # 1. DISCOVERY
        for root, dirs, files in os.walk(ROOT_PHOTO_DIR):
            profile_path = os.path.join(root, "profile.json")
            local_cache = {}
            if os.path.exists(profile_path):
                try: 
                    with open(profile_path, 'r') as f: local_cache = json.load(f)
                except: pass
            
            updates_by_dir[root] = local_cache
            
            for file in files:
                if os.path.splitext(file)[1].lower() in ALLOWED_EXTENSIONS:
                    all_files_count += 1
                    full_path = os.path.join(root, file)
                    
                    should_process = True
                    
                    if file in local_cache:
                        c = local_cache[file]
                        # CRITICAL FIX: Check if data is legacy (missing 'id')
                        # If legacy, we force re-process to generate IDs and fix coordinates
                        if c.get("faces") and "id" not in c["faces"][0]:
                            should_process = True
                        else:
                            should_process = False
                            # Hydrate
                            self.ingest_data(full_path, c["md5"], c["faces"], c.get("created"), c.get("modified"))

                    if should_process:
                        files_to_process.append((full_path, root))

        global_state["status"]["total_files"] = all_files_count
        
        # 2. PROCESSING
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
                        self.ingest_data(res["full_path"], res["md5"], res["faces"], res["created"], res["modified"])
                        updates_by_dir[res["root_dir"]][res["filename"]] = {
                            "md5": res["md5"], "created": res["created"], "modified": res["modified"], "faces": res["faces"]
                        }

        # Save all profiles
        for root, data in updates_by_dir.items():
            if data:
                with open(os.path.join(root, "profile.json"), 'w') as f: json.dump(data, f, indent=4)

        # 3. CLUSTERING
        self.perform_clustering(updates_by_dir)
        
        global_state["status"].update({"is_scanning": False, "progress": 100, "current_task": "Done"})

    def ingest_data(self, full_path, md5, faces, c_time, m_time):
        global_state["path_map"][full_path] = md5
        folder = os.path.basename(os.path.dirname(full_path))
        parts = folder.split('_')
        location = "_".join(parts[1:]) if len(parts) > 1 else folder

        # Safety: Backfill IDs if missing (should be handled by re-scan logic, but double safety)
        for f in faces:
            if "id" not in f and "box" in f:
                 b = f["box"]
                 f["id"] = f"{b[0]}_{b[1]}_{b[2]}_{b[3]}"

        if md5 not in global_state["photos"]:
            global_state["photos"][md5] = {
                "paths": [], "faces": faces, "location": location,
                "folder": folder, "created": c_time, "modified": m_time, 
                "filename": os.path.basename(full_path)
            }
        
        if full_path not in global_state["photos"][md5]["paths"]:
            global_state["photos"][md5]["paths"].append(full_path)

    def perform_clustering(self, updates_by_dir):
        global_state["status"]["current_task"] = "Clustering..."
        
        # 1. Prepare Data
        # We need to map the flat list of encodings back to specific faces
        all_encodings = []
        map_back = [] # List of (md5, face_index)
        
        for md5, pdata in global_state["photos"].items():
            for i, face in enumerate(pdata["faces"]):
                # CRITICAL: We DO NOT cluster faces that are:
                # 1. Manual (User confirmed)
                # 2. Deleted (Soft deleted faces should stay deleted)
                if not face.get("manual", False) and not face.get("is_deleted", False):
                    # Validate encoding
                    enc = face["encoding"]
                    if len(enc) == 512 and not np.isnan(np.sum(enc)):
                        all_encodings.append(enc)
                        map_back.append((md5, i))
        
        if not all_encodings: return

        print(f"Clustering {len(all_encodings)} faces...")
        
        # 2. Run DBSCAN
        clt = DBSCAN(metric="cosine", n_jobs=-1, eps=0.40, min_samples=1)
        clt.fit(all_encodings)
        
        # 3. Group Results by Temporary Label
        # temp_clusters = { "0": [(md5, idx), ...], "1": [...] }
        temp_clusters = {}
        for i, label_id in enumerate(clt.labels_):
            if label_id == -1: continue # Noise
            str_label = str(label_id)
            if str_label not in temp_clusters: temp_clusters[str_label] = []
            temp_clusters[str_label].append(map_back[i])

        dirty_dirs = set()

        # 4. Assign Persistent UUIDs
        for label, face_locations in temp_clusters.items():
            
            # Step A: Check if this group belongs to an existing Person
            # We look at all faces in this new group. If they previously had a valid cluster_id (UUID),
            # we vote. (e.g., if 80% of faces were 'Person-A', this group is 'Person-A')
            
            existing_ids = []
            for (md5, f_idx) in face_locations:
                current_cid = global_state["photos"][md5]["faces"][f_idx]["cluster_id"]
                if current_cid != "-1" and current_cid in global_state["people"]:
                    existing_ids.append(current_cid)
            
            final_person_id = None
            
            if existing_ids:
                # Find most common existing ID (Stability check)
                from collections import Counter
                most_common = Counter(existing_ids).most_common(1)
                if most_common:
                    final_person_id = most_common[0][0]
            
            # Step B: If no existing ID found, create NEW UUID
            if not final_person_id:
                final_person_id = uuid.uuid4().hex # Unique 32-char string
                
                # Create Profile
                # Use the first face as avatar
                first_md5, first_idx = face_locations[0]
                first_face = global_state["photos"][first_md5]["faces"][first_idx]
                
                global_state["people"][final_person_id] = {
                    "name": f"Person {label}", # Display name can use label temporarily
                    "gender": "Unknown",
                    "avatar_md5": first_md5, 
                    "avatar_face_id": first_face.get("id"),
                    "is_deleted": False # Init flag
                }
                save_people_db()

            # Step C: Apply this UUID to all faces in the cluster
            for (md5, f_idx) in face_locations:
                face = global_state["photos"][md5]["faces"][f_idx]
                
                # Only update if changed
                if face["cluster_id"] != final_person_id:
                    face["cluster_id"] = final_person_id
                    
                    # Mark for disk save
                    if global_state["photos"][md5]["paths"]:
                        main_path = global_state["photos"][md5]["paths"][0]
                        d = os.path.dirname(main_path)
                        f = os.path.basename(main_path)
                        
                        if d in updates_by_dir and f in updates_by_dir[d]:
                            # Ensure index validity
                            if f_idx < len(updates_by_dir[d][f]["faces"]):
                                updates_by_dir[d][f]["faces"][f_idx]["cluster_id"] = final_person_id
                                dirty_dirs.add(d)

        # 5. Save Disk Updates
        for d in dirty_dirs:
            p_path = os.path.join(d, "profile.json")
            with open(p_path, 'w') as f: json.dump(updates_by_dir[d], f, indent=4)
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
        if f not in res: res[f] = {"name": f, "location": p["location"], "count": 0}
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
                "md5": md5,
                "url": f"/image_by_md5/{md5}",
                "filename": p["filename"],
                "faces": faces_simple
            })
    return jsonify(res)

@app.route('/api/people')
def get_people():
    res = []
    counts = {}
    for md5, p in global_state["photos"].items():
        for f in p["faces"]:
            # Don't count deleted faces
            if not f.get("is_deleted", False):
                counts[f["cluster_id"]] = counts.get(f["cluster_id"], 0) + 1
            
    for cid, p in global_state["people"].items():
        if cid == "-1": continue
        # FILTER: Don't show deleted people
        if p.get("is_deleted", False): continue
        
        # FILTER: Don't show people with 0 photos (cleaned up automatically)
        if counts.get(cid, 0) == 0: continue

        res.append({
            "id": cid, 
            "name": p["name"], 
            "gender": p["gender"],
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

# --- IMAGE SERVING ---

@app.route('/image_by_md5/<md5>')
def serve_img(md5):
    if md5 in global_state["photos"] and global_state["photos"][md5]["paths"]:
        return send_file(global_state["photos"][md5]["paths"][0])
    return "Missing", 404

@app.route('/api/face_crop/<md5>/<face_id>')
def serve_crop(md5, face_id):
    if md5 not in global_state["photos"]: return "Err", 404
    
    target_face = None
    for f in global_state["photos"][md5]["faces"]:
        if f["id"] == face_id:
            target_face = f
            break
            
    if not target_face: return "Face Not Found", 404
    
    path = global_state["photos"][md5]["paths"][0]
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img) 
        
        top, right, bottom, left = target_face["box"]
        pad = 50
        w, h = img.size
        crop = img.crop((max(0, left-pad), max(0, top-pad), min(w, right+pad), min(h, bottom+pad))).convert("RGB")
        
        bio = io.BytesIO()
        crop.save(bio, 'JPEG')
        bio.seek(0)
        return send_file(bio, mimetype='image/jpeg')
    except:
        import traceback
        traceback.print_exc()
        return "Err", 500

@app.route('/api/photo_details/<md5>')
def details(md5):
    if md5 not in global_state["photos"]: return jsonify({}), 404
    p = global_state["photos"][md5]
    
    exif = {}
    gps = None
    try:
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
        faces.append({
            "name": global_state["people"][cid]["name"] if cid in global_state["people"] else "Unknown",
            "face_id": f["id"], "md5": md5
        })

    return jsonify({
        "filename": p["filename"], "folder": p["folder"], "location": p["location"],
        "gps": gps, "created": p["created"], "exif": exif, "faces": faces, "full_path_id": md5 
    })

# --- DATA MANIPULATION ---

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
    updates_by_dir = {}
    
    # 1. Soft Delete the Person Profile
    for i in ids: 
        if i in global_state["people"]: 
            global_state["people"][i]["is_deleted"] = True
    save_people_db()

    # 2. Mark faces as deleted (so they are ignored in future scans)
    # We do NOT set cluster_id to "-1". We keep the link but flag it.
    # This prevents the AI from picking them up as "New Person" next time.
    
    for md5, p in global_state["photos"].items():
        changed = False
        for i, f in enumerate(p["faces"]):
            if f["cluster_id"] in ids:
                f["is_deleted"] = True   # NEW FLAG
                f["manual"] = True       # Lock it
                f["cluster_id"] = "-1"   # Visually remove from group
                
                # Prepare disk update
                if p["paths"]:
                    d = os.path.dirname(p["paths"][0])
                    fn = p["filename"]
                    if d not in updates_by_dir: updates_by_dir[d] = {}
                    if fn not in updates_by_dir[d]: updates_by_dir[d][fn] = {}
                    # We need to know which face index to update. 
                    # Ideally we load the file content here, but let's assume batched save below
                    updates_by_dir[d][fn][i] = True # Mark index as needing update

        if changed: save_profile_for_md5(md5)

    # 3. Batch Save to Disk (Optimized)
    # We need to iterate the updates_by_dir properly
    for d, file_map in updates_by_dir.items():
        p_path = os.path.join(d, "profile.json")
        if os.path.exists(p_path):
            try:
                with open(p_path, 'r') as f: data = json.load(f)
                write_needed = False
                for fname, indices in file_map.items():
                    if fname in data:
                        for idx in indices:
                            if idx < len(data[fname]['faces']):
                                data[fname]['faces'][idx]["is_deleted"] = True
                                data[fname]['faces'][idx]["manual"] = True
                                data[fname]['faces'][idx]["cluster_id"] = "-1"
                                write_needed = True
                if write_needed:
                    with open(p_path, 'w') as f: json.dump(data, f, indent=4)
            except: pass

    return jsonify({"success": True})

@app.route('/api/update_person', methods=['POST'])
def update_person():
    d = request.json
    cid = d.get('id')
    if cid in global_state["people"]:
        global_state["people"][cid].update({"name": d.get('name'), "gender": d.get('gender')})
        save_people_db()
        return jsonify({"success": True})
    return jsonify({"success": False})

@app.route('/api/merge_people', methods=['POST'])
def merge_people():
    d = request.json
    sid = d.get('source_id')
    tid = d.get('target_id')
    
    if sid not in global_state["people"] or tid not in global_state["people"]:
        return jsonify({"success": False})

    for md5, p in global_state["photos"].items():
        changed = False
        for f in p["faces"]:
            if f["cluster_id"] == sid:
                f["cluster_id"] = tid
                f["manual"] = True
                changed = True
        if changed: save_profile_for_md5(md5)

    if sid in global_state["people"]: del global_state["people"][sid]
    save_people_db()
    return jsonify({"success": True})

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
                # Match by face ID
                for mf in mem_faces:
                    for df in data[fname]["faces"]:
                        # If legacy format on disk without ID, rely on box match
                        if "id" in df:
                            if df["id"] == mf["id"]:
                                df["cluster_id"] = mf["cluster_id"]
                                df["manual"] = mf["manual"]
                        elif df["box"] == mf["box"]:
                                df["cluster_id"] = mf["cluster_id"]
                                df["manual"] = mf["manual"]
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
    if not query:
        return jsonify([])
    
    results = []
    
    # Pre-fetch people names for faster lookup
    # { cluster_id: name_lowercase }
    people_map = {cid: p['name'].lower() for cid, p in global_state['people'].items() if not p.get('is_deleted')}

    for md5, p in global_state["photos"].items():
        match = False
        
        # 1. Check Metadata (Filename, Location, Date)
        # Date format in memory is usually "YYYY-MM-DD HH:MM:SS"
        if (query in p['filename'].lower() or 
            query in p['location'].lower() or 
            query in p['folder'].lower() or
            query in p['created']): # Matches "2023", "2023-12", "12-25"
            match = True
            
        # 2. Check People in Photo
        if not match:
            for face in p['faces']:
                cid = face['cluster_id']
                if cid in people_map and query in people_map[cid]:
                    match = True
                    break
        
        if match:
            # Format exactly like photos_by_dir
            faces_simple = []
            for face in p["faces"]:
                cid = face["cluster_id"]
                name = global_state["people"][cid]["name"] if cid in global_state["people"] else "Unknown"
                faces_simple.append({"name": name, "face_id": face["id"], "md5": md5})
            
            results.append({
                "md5": md5,
                "url": f"/image_by_md5/{md5}",
                "filename": p["filename"],
                "created": p["created"], # Added for sort
                "faces": faces_simple
            })

    # Sort results by date (newest first)
    results.sort(key=lambda x: x['created'], reverse=True)
    return jsonify(results)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    load_people_db()
    app.run(debug=True, port=args.port)