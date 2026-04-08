import cv2
import os

# ===============================
# Function to extract frames
# ===============================
def extract_frames(video_path, save_path, fps=5):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps == 0:
        print(f"❌ FPS is 0 for video: {video_path}")
        return

    frame_interval = int(original_fps / fps)
    if frame_interval == 0:
        frame_interval = 1

    os.makedirs(save_path, exist_ok=True)

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_path = os.path.join(save_path, f"{saved}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"✅ Extracted {saved} frames from {video_path}")


# ===============================
# MAIN SCRIPT
# ===============================

raw_path = "dataset_raw"
save_root = "dataset_frames"

# Make sure output folder exists
os.makedirs(save_root, exist_ok=True)

for label in ["pain","no_pain"]:

    class_path = os.path.join(raw_path, label)

    if not os.path.exists(class_path):
        print(f"⚠️ Folder not found: {class_path}")
        continue

    for file in os.listdir(class_path):

        if not file.lower().endswith(".mp4"):
            continue  # skip non-video files

        video_path = os.path.join(class_path, file)

        video_name = os.path.splitext(file)[0]

        save_path = os.path.join(save_root, label, video_name)

        extract_frames(video_path, save_path)

print("\n🎉 Done extracting all videos.")
