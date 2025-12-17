### 1. `profile.json` (The Physical Evidence)

*   **Location:** One `profile.json` file is created **inside every single photo subdirectory** (e.g., `C:/Photos/2023_Tokyo/profile.json`). This is called a "sidecar" file.
*   **Purpose:** To store all the AI-generated data about the images **in that specific folder**. It acts as a cache. If you move the folder, the AI data moves with it.
*   **Key:** The top-level key is the **filename** of the image (e.g., `"IMG_1234.JPG"`).

**Structure:**

```json
{
  "IMG_1234.JPG": {
    "md5": "a1b2c3d4...",
    "created": "2023-12-01 10:30:00",
    "modified": "2023-12-02 11:00:00",
    "is_video": false,
    "faces": [
      {
        "id": "123_456_789_101",
        "box": [123, 456, 789, 101],
        "encoding": [-0.12, 0.45, ...],
        "cluster_id": "a1b2c3d4-e5f6-...",
        "manual": false,
        "is_deleted": false
      },
      {
        "id": "500_600_700_800",
        "box": [500, 600, 700, 800],
        "encoding": [0.98, -0.23, ...],
        "cluster_id": "-1",
        "manual": true,
        "is_deleted": true
      }
    ]
  },
  "VIDEO_001.MP4": {
    "md5": "e5f6g7h8...",
    "created": "...",
    "modified": "...",
    "is_video": true,
    "faces": []
  }
}
```

*   **`md5`**: The unique fingerprint of the file. If this changes, the system knows the file has been edited and re-scans it.
*   **`faces`**: An array containing every face found in that specific image.
    *   **`id`**: A unique ID for the face *within this image*, generated from its coordinates. This is the **stable key** we use to identify a face for cropping.
    *   **`box`**: The raw `[top, right, bottom, left]` coordinates.
    *   **`encoding`**: The 512 numbers that represent the "face vector". This is the heavy AI data.
    *   **`cluster_id`**: This is the **link** to `people.json`. It stores the UUID of the person this face belongs to. `-1` means "Unknown".
    *   **`manual`**: `true` if you have ever manually edited this face (assigned, hidden, merged). This "locks" it from being changed by the automatic clustering.
    *   **`is_deleted`**: `true` if you have hidden this specific face.

---

### 2. `people.json` (The Logical Identity)

*   **Location:** **Only one** `people.json` file in the root directory where `app.py` runs.
*   **Purpose:** To store the "Profiles" of every unique person the system knows about. It defines who a person is, not where they are.
*   **Key:** The top-level key is the **UUID** of the person (the same `cluster_id` from `profile.json`).

**Structure:**

```json
{
  "a1b2c3d4-e5f6-...": {
    "name": "Alice",
    "gender": "Female",
    "avatar_md5": "a1b2c3d4...",
    "avatar_face_id": "123_456_789_101",
    "is_deleted": false
  },
  "f7g8h9i0-j1k2-...": {
    "name": "Bob (Deleted)",
    "gender": "Male",
    "avatar_md5": "x1y2z3...",
    "avatar_face_id": "...",
    "is_deleted": true
  },
  "k3l4m5n6-o7p8-...": {
    "name": "New Person 1a2b3c",
    "gender": "Unknown",
    "avatar_md5": "y4z5a6...",
    "avatar_face_id": "...",
    "is_deleted": false
  }
}
```

*   **`name`**: The display name you give them (e.g., "Alice").
*   **`gender`**: "Male", "Female", or "Unknown".
*   **`avatar_md5`**: The MD5 of the **image** that contains the profile picture.
*   **`avatar_face_id`**: The specific ID of the **face** within that image to use as the profile picture. When you click the "Star" icon, you are updating these two values.
*   **`is_deleted`**: `true` if you have deleted this entire person. The system will keep their entry here (to remember not to rediscover them) but will hide them from the UI.

### The Workflow Connection
1.  When you open a photo, the app reads `profile.json` to find the faces.
2.  It takes `cluster_id: "a1b2c3d4-e5f6-..."` from a face.
3.  It looks up `"a1b2c3d4-e5f6-..."` in `people.json`.
4.  It finds `name: "Alice"` and displays "Alice" next to the face.