# AI Photo Manager
**Your memories, organized.**

Stop digging through endless folders and cryptic filenames. **AI Photo Manager** transforms your local collection of photos and videos into a smart, beautiful, and searchable library—running entirely on your own computer. No cloud uploads, no privacy concerns, just your memories.

It organizes your life into three simple views:

### 👥 People
Forgot where you saved those photos of your best friend? The built-in AI automatically detects faces and groups them together.
*   **Identify** unknown faces with a click.
*   **Merge** duplicates easily.
*   **Search** your entire library just by clicking a person's face.

### 🌍 Location
Relive your journeys on an interactive world map.
*   **See the World:** Your photos are automatically grouped into dynamic clusters, showing you exactly where you've been.
*   **Travel Timeline:** Jump from Paris to Tokyo instantly by browsing your travel history chronologically directly on the map.

### 📸 Albums
Browse your folders as beautiful, high-performance visual grids.
*   **Seamless Playback:** Videos play instantly right alongside your photos.
*   **Smart Metadata:** Your edits and AI data are stored locally in the folder, so your library is always portable and safe.

## 🛠️ Prerequisites

*   **Python 3.8+**
*   **Visual C++ Redistributable** (Required for OpenCV/InsightFace on Windows)

## 📦 Installation

1.  **Clone or Download this repository.**

2.  **Install Python Dependencies:**
    Create a `requirements.txt` file (or just run the command below) with the following packages:
    ```txt
    flask
    opencv-python
    insightface
    onnxruntime
    pillow
    scikit-learn
    numpy
    ```

    Run the installation:
    ```bash
    pip install -r requirements.txt
    ```

    > **Note:** If you don't have a dedicated GPU, `onnxruntime` (CPU) is sufficient. For faster processing with NVIDIA cards, install `onnxruntime-gpu`.

3.  **Setup Credentials:**
    The system will automatically generate a `settings.json` file on the first run, but you can create it manually to set your password immediately:

    *Create a file named `settings.json` in the root directory:*
    ```json
    {
       "users": {
          "admin": "your_password_here"
       }
    }
    ```

## 🖥️ Usage

1.  **Start the Server:**
    Run the application by pointing it to your main photo directory.

    ```bash
    python app.py --root-photo-dir "C:/Path/To/My/Photos"
    ```

2.  **Access the Web Interface:**
    Open your browser and navigate to:
    `http://localhost:5000`

3.  **Login:**
    Use the username `admin` and the password you set in `settings.json` (default is `pass`).

4.  **Scan Library:**
    Click the **"🔄 Refresh"** button in the sidebar to start scanning photos, detecting faces, and extracting GPS data.

## 📂 Data Structure

This application does not use a central database for photo metadata. Instead, it uses **Sidecar Files**:

*   **`profile.json`**: Created inside *each* photo subdirectory. Contains face encodings, bounding boxes, and image hashes for that specific folder.
*   **`people.json`**: Located in the application root. Maps unique UUIDs to Person Names and Avatars.

## 🤝 Face Management Workflow

1.  **Scan:** Wait for the scanner to finish.
2.  **Identify:** Go to an album or photo. Click on an "Unknown" face (dashed border).
3.  **Name:** Enter a name to create a new person.
4.  **Merge:** If the system finds the same person but thinks they are new, click the face, select the existing name from the dropdown, and click "Save" to merge them.

## 📄 License

[MIT License](LICENSE) (or whichever license you prefer)
