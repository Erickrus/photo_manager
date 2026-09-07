import json
import os
import sys
import datetime
import argparse
import subprocess

# Path to livetext.py, used as an isolated subprocess. Running OCR in a child
# process means an uncatchable ObjC/Vision crash kills only that child — the
# batch keeps going instead of the whole program dying.
_LIVETEXT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "livetext.py")


def extract_text_isolated(image_path, timeout=180):
    """Run livetext.extract_text in a subprocess so a native crash can't take
    down this process. Returns extracted text, or "" on any failure.

    Raises subprocess.TimeoutExpired only internally (caught here).
    """
    try:
        result = subprocess.run(
            [sys.executable, _LIVETEXT_SCRIPT, image_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"    Timed out after {timeout}s → skipping")
        return ""

    if result.returncode != 0:
        # Child crashed (e.g. Vision ObjC exception) or errored. Log briefly.
        err = (result.stderr or "").strip().splitlines()
        tail = err[-1] if err else f"exit code {result.returncode}"
        print(f"    OCR subprocess failed: {tail}")
        return ""

    return result.stdout


def process_profile_with_ocr(profile_path: str, refresh: bool = False):
    profile_dir = os.path.dirname(profile_path)

    print(f"\n{datetime.datetime.now()} - Loading: {profile_path}")

    try:
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile = json.load(f)
    except Exception as e:
        print(f"  Failed to load JSON: {e}")
        return

    # === Check if already processed by apple-livetext ===
    # In refresh mode we re-run regardless and drop old apple-livetext comments.
    processors = profile.get("processor", [])
    if "apple-livetext" in processors and not refresh:
        print("  Already processed by apple-livetext → Skipping")
        return

    # === Proceed with OCR ===
    allowed_ext = {'.jpg', '.jpeg', '.png'}
    updated = False

    for filename, metadata in profile.items():
        ext = os.path.splitext(filename)[1].lower()
        if ext not in allowed_ext or metadata.get("is_video", False):
            continue

        full_path = os.path.join(profile_dir, filename)
        if not os.path.exists(full_path):
            print(f"  Warning: Image not found: {full_path}")
            continue

        print(f"  {datetime.datetime.now()} → Processing {filename}")

        # In refresh mode, drop any prior apple-livetext comments up front so
        # stale OCR is removed even if this image now yields no text.
        comments = metadata.get("comments", [])
        if refresh:
            cleaned = [c for c in comments if c.get("author") != "apple-livetext"]
            if len(cleaned) != len(comments):
                comments = cleaned
                metadata["comments"] = comments
                updated = True
                print(f"    Removed stale apple-livetext comment(s)")

        try:
            ocr_text = extract_text_isolated(full_path).strip()

            if not ocr_text:
                print(f"    No text detected")
                continue

            print(f"    OCR result ({len(ocr_text)} chars): {ocr_text[:100]}...")

            new_comment = {
                "text": ocr_text,
                "author": "apple-livetext",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Avoid duplicate comments (extra safety)
            if not any(c["author"] == "apple-livetext" for c in comments):
                comments.append(new_comment)
                metadata["comments"] = comments
                updated = True
                print(f"    Comment added")

        except Exception as e:
            print(f"    Error processing {filename}: {e}")

    # === Save changes and mark as processed ===
    if updated:
        # Add or update the processor list
        if "processor" not in profile:
            profile["processor"] = []
        if "apple-livetext" not in profile["processor"]:
            profile["processor"].append("apple-livetext")

        try:
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=4, ensure_ascii=False)
            print(f"  Updated + marked as processed: {profile_path}")
        except Exception as e:
            print(f"  Failed to save: {e}")
    else:
        # Optional: still mark as processed even if no text found?
        # Here we choose NOT to mark if nothing was added (you can change this)
        print("  No new OCR text found → No changes made")


# ------------------------------------------------------------------
# Run it
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process all profiles in a photo directory with Apple LiveText OCR")
    parser.add_argument('--root-photo-dir', required=True, help='Root directory containing profile.json files')
    parser.add_argument('--refresh', action='store_true', default=False,
                        help='Re-run OCR even on already-processed profiles, replacing prior apple-livetext comments with fresh results')
    args = parser.parse_args()

    root_dir = args.root_photo_dir

    if not os.path.exists(root_dir):
        print(f"Error: Directory not found: {root_dir}")
        exit(1)

    print(f"Scanning for profiles in: {root_dir}")
    profiles_found = []

    for root, dirs, files in os.walk(root_dir):
        if "profile.json" in files:
            profile_path = os.path.join(root, "profile.json")
            profiles_found.append(profile_path)

    if not profiles_found:
        print("No profile.json files found")
        exit(0)

    print(f"Found {len(profiles_found)} profile(s)")

    if args.refresh:
        print("Refresh mode: re-running all profiles and replacing prior apple-livetext comments")

    for idx, profile_path in enumerate(profiles_found, 1):
        print(f"\n[{idx}/{len(profiles_found)}] Processing: {profile_path}")
        process_profile_with_ocr(profile_path, refresh=args.refresh)