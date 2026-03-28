import os
import json
import torch
import re
import time
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
import cv2

# ---------------- CONFIG ----------------
CLASS_NAMES = [
    'Biopsy',
    'Blood_due_to_biopsy',
    'Blurred_Dark',
    'Close_to_mucosa',
    'Readable',
    'Stool'
]

BASE_DIR = Path(__file__).resolve().parent
DEVICE = "cpu"
READABLE_IDX = CLASS_NAMES.index("Readable")

BATCH_SIZE = 2
FRAME_SKIP = 4
MAX_BATCH_SIZE = 4 * 1024 * 1024 * 1024  # 4GB

torch.set_num_threads(torch.get_num_threads())
print(f"Using {torch.get_num_threads()} CPU thread(s).")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
])

# ---------------- UTILS ----------------
def sanitize_label(label: str) -> str:
    label = label.strip()
    label = label.replace('/', '_')
    label = re.sub(r'[\\:*?"<>|]', '_', label)
    label = label.replace(' ', '_')
    return label

def parse_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    frame_labels = {}
    for item in data.get("items", []):
        frame_id = item.get("attr", {}).get("frame")
        if frame_id is None:
            continue

        label = None
        for ann in item.get("annotations", []):
            attrs = ann.get("attributes", {})
            if "Frame Quality" in attrs:
                label = sanitize_label(attrs["Frame Quality"])
                break

        if label:
            frame_labels[int(frame_id)] = label

    return frame_labels

def get_file_size_bytes(frame):
    success, encoded = cv2.imencode('.jpg', frame)
    if not success:
        return 0
    return len(encoded)

# ---------------- MODEL ----------------
def load_model(model_path):
    model = torch.jit.load(model_path, map_location=DEVICE)
    model.eval()
    return model

# ---------------- INFERENCE ----------------
def infer_batch(model, batch_tensor):
    with torch.inference_mode():
        outputs_orig = model(batch_tensor)
        outputs_flip = model(torch.flip(batch_tensor, dims=[3]))
        outputs = (outputs_orig + outputs_flip) / 2

        outputs[:, READABLE_IDX] += 0.0

        probs = torch.softmax(outputs, dim=1)
        pred = outputs.argmax(1)

        pred = torch.where(
            probs[:, READABLE_IDX] > 0.3,
            torch.full_like(pred, READABLE_IDX),
            pred
        )

    return pred.cpu().numpy(), probs.cpu().numpy()

# ---------------- MAIN ----------------
def run(video_path, json_path, model_path, output_dir):
    start_time = time.time()

    gt_labels = parse_json(json_path)
    model = load_model(model_path)

    video_name = video_path.stem
    base_output = Path(output_dir) / video_name

    current_batch = 1
    current_batch_size = 0
    batching_enabled = False

    def get_dirs():
        if not batching_enabled:
            correct_dir = base_output / "correct"
            incorrect_dir = base_output / "incorrect"
        else:
            batch_dir = base_output / f"batch{current_batch}"
            correct_dir = batch_dir / "correct"
            incorrect_dir = batch_dir / "incorrect"

        for cls in CLASS_NAMES:
            os.makedirs(correct_dir / cls, exist_ok=True)

        return correct_dir, incorrect_dir

    correct_dir, incorrect_dir = get_dirs()

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    buffer = []
    frame_indices = []
    original_frames = []

    total = correct = 0
    tp_readable = fp_readable = fn_readable = 0
    total_infer_time = 0

    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")
    frame_idx = 0

    def process_batch(preds, frame_indices, original_frames):
        nonlocal total, correct
        nonlocal tp_readable, fp_readable, fn_readable
        nonlocal current_batch, current_batch_size, batching_enabled
        nonlocal correct_dir, incorrect_dir

        for i, pred_idx in enumerate(preds):
            idx = frame_indices[i]

            if idx not in gt_labels:
                continue

            gt = gt_labels[idx]
            pred = CLASS_NAMES[pred_idx]

            filename = f"frame_{idx:06d}.jpg"

            total += 1

            if pred == gt:
                dest = correct_dir / gt / filename
                correct += 1
            else:
                dest = incorrect_dir / f"{gt}_pred_{pred}" / filename
                dest.parent.mkdir(parents=True, exist_ok=True)

            # Readable metrics
            if pred == "Readable":
                if gt == "Readable":
                    tp_readable += 1
                else:
                    fp_readable += 1
            if gt == "Readable" and pred != "Readable":
                fn_readable += 1

            frame = original_frames[i]
            frame_size = get_file_size_bytes(frame)

            # ---- BATCHING LOGIC ----
            if not batching_enabled:
                if current_batch_size + frame_size > MAX_BATCH_SIZE:
                    batching_enabled = True
                    current_batch = 2
                    current_batch_size = 0
                    correct_dir, incorrect_dir = get_dirs()
                    print(f"Exceeded 4GB → switching to batch2")

            else:
                if current_batch_size + frame_size > MAX_BATCH_SIZE:
                    current_batch += 1
                    current_batch_size = 0
                    correct_dir, incorrect_dir = get_dirs()
                    print(f"Switching to batch{current_batch}")

            current_batch_size += frame_size
            cv2.imwrite(str(dest), frame)


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            pbar.update(1)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb)

        buffer.append(tensor)
        frame_indices.append(frame_idx)
        original_frames.append(frame)

        if len(buffer) == BATCH_SIZE:
            start_inf = time.time()

            batch_tensor = torch.stack(buffer).to(DEVICE)
            preds, _ = infer_batch(model, batch_tensor)

            total_infer_time += time.time() - start_inf

            process_batch(preds, frame_indices, original_frames)

            buffer, frame_indices, original_frames = [], [], []

        frame_idx += 1
        pbar.update(1)

    cap.release()

    # leftover
    if buffer:
        batch_tensor = torch.stack(buffer).to(DEVICE)
        preds, _ = infer_batch(model, batch_tensor)
        process_batch(preds, frame_indices, original_frames)

    # --------- METRICS ---------
    accuracy = correct / total if total else 0
    precision = tp_readable / (tp_readable + fp_readable) if (tp_readable + fp_readable) else 0
    recall = tp_readable / (tp_readable + fn_readable) if (tp_readable + fn_readable) else 0

    total_time = time.time() - start_time

    report_path = base_output / "report.txt"
    with open(report_path, "w") as f:
        f.write(f"Video: {video_name}\n")
        f.write(f"Total evaluated: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Readable Metrics:\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n\n")
        f.write(f"Total time (sec): {total_time:.2f}\n")
        f.write(f"Inference time (sec): {total_infer_time:.2f}\n")

    print(f"\nTotal pipeline time: {total_time:.2f} sec")
    print(f"Total inference time: {total_infer_time:.2f} sec")
    print("DONE")

# ---------------- MULTI VIDEO ----------------
def run_on_folder(video_dir, model_path, output_dir):
    video_dir = Path(video_dir)

    video_files = [
        f for f in video_dir.iterdir()
        if f.suffix.lower() in [".mp4", ".avi"]
    ]

    if len(video_files) == 0:
        print("No videos found.")
        return

    print(f"Found {len(video_files)} videos.\n")

    overall_start = time.time()

    for i, video_path in enumerate(video_files, 1):
        video_name = video_path.stem
        json_path = video_dir / video_name / "Train.json"

        print(f"\n[{i}/{len(video_files)}] Processing: {video_name}")

        if not json_path.exists():
            print(f"Skipping {video_name} (Train.json not found)")
            continue

        try:
            run(video_path, json_path, model_path, output_dir)
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            continue

    total_time = time.time() - overall_start
    print(f"\nAll videos processed in {total_time:.2f} sec")

# ---------------- RUN ----------------
if __name__ == "__main__":
    VIDEO_DIR = BASE_DIR.parent / "model validation 2"
    MODEL_PATH = BASE_DIR / "model.pt"
    OUTPUT_DIR = BASE_DIR / "results"

    run_on_folder(VIDEO_DIR, MODEL_PATH, OUTPUT_DIR)
