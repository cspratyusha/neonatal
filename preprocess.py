import cv2
import os


def extract_frames(video_path, save_path, fps=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        print(f"❌ FPS=0 for: {video_path}")
        cap.release()
        return

    frame_interval = max(1, int(original_fps / fps))
    os.makedirs(save_path, exist_ok=True)

    count = saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            # Save as PNG for lossless quality — avoids JPEG compression artifacts
            cv2.imwrite(os.path.join(save_path, f"{saved:05d}.png"), frame)
            saved += 1
        count += 1

    cap.release()
    print(f"✅ {saved} frames → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
raw_path  = "dataset_raw"
save_root = "dataset_frames"
os.makedirs(save_root, exist_ok=True)

for label in ["pain", "no_pain"]:
    class_path = os.path.join(raw_path, label)
    if not os.path.exists(class_path):
        print(f"⚠️  Not found: {class_path}")
        continue

    for file in sorted(os.listdir(class_path)):
        if not file.lower().endswith((".mp4", ".avi", ".mov")):
            continue
        video_name = os.path.splitext(file)[0]
        extract_frames(
            video_path=os.path.join(class_path, file),
            save_path=os.path.join(save_root, label, video_name),
        )

print("\n🎉 Preprocessing complete.")