import os
import torch
import time
import cv2
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

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
print(f"Using {torch.get_num_threads()} CPU thread.")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
])

# ---------------- UTILS ----------------
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

    return pred.cpu().numpy()

# ---------------- MAIN ----------------
def run(video_path, model_path, output_dir):
    start_time = time.time()

    video_path = Path(video_path)
    video_name = video_path.stem

    model = load_model(model_path)

    base_output = Path(output_dir) / video_name

    # ---- BATCH CONTROL ----
    current_batch = 1
    current_batch_size = 0
    batching_enabled = False

    def get_dirs():
        if not batching_enabled:
            base = base_output
        else:
            base = base_output / f"batch{current_batch}"

        for cls in CLASS_NAMES:
            os.makedirs(base / cls, exist_ok=True)

        return base

    current_base = get_dirs()

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    buffer = []
    original_frames = []
    frame_indices = []

    class_counts = {cls: 0 for cls in CLASS_NAMES}
    total_infer_time = 0

    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")

    frame_idx = 0

    def process_batch(preds, frame_indices, original_frames):
        nonlocal current_batch, current_batch_size, batching_enabled
        nonlocal current_base

        for i, pred_idx in enumerate(preds):
            pred = CLASS_NAMES[pred_idx]
            filename = f"frame_{frame_indices[i]:06d}.jpg"

            frame = original_frames[i]
            frame_size = get_file_size_bytes(frame)

            # ---- BATCHING LOGIC ----
            if not batching_enabled:
                if current_batch_size + frame_size > MAX_BATCH_SIZE:
                    batching_enabled = True
                    current_batch = 2
                    current_batch_size = 0
                    current_base = get_dirs()
                    print("Exceeded 4GB → switching to batch2")
            else:
                if current_batch_size + frame_size > MAX_BATCH_SIZE:
                    current_batch += 1
                    current_batch_size = 0
                    current_base = get_dirs()
                    print(f"Switching to batch{current_batch}")

            current_batch_size += frame_size

            dest = current_base / pred / filename
            cv2.imwrite(str(dest), frame)

            class_counts[pred] += 1

    # -------- LOOP --------
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
        original_frames.append(frame)
        frame_indices.append(frame_idx)

        if len(buffer) == BATCH_SIZE:
            start_inf = time.time()

            batch_tensor = torch.stack(buffer).to(DEVICE)
            preds = infer_batch(model, batch_tensor)

            total_infer_time += time.time() - start_inf

            process_batch(preds, frame_indices, original_frames)

            buffer, original_frames, frame_indices = [], [], []

        frame_idx += 1
        pbar.update(1)

    cap.release()

    # leftover
    if buffer:
        batch_tensor = torch.stack(buffer).to(DEVICE)
        preds = infer_batch(model, batch_tensor)
        process_batch(preds, frame_indices, original_frames)

    total_time = time.time() - start_time

    # -------- REPORT --------
    report_path = base_output / "report.txt"
    with open(report_path, "w") as f:
        f.write(f"Video: {video_name}\n")
        f.write(f"Total frames classified: {sum(class_counts.values())}\n\n")
        f.write("Class Distribution:\n")
        for cls in CLASS_NAMES:
            f.write(f"{cls}: {class_counts[cls]}\n")
        f.write(f"\nTotal pipeline time (sec): {total_time:.2f}\n")
        f.write(f"Inference time (sec): {total_infer_time:.2f}\n")

    print(f"\nTotal pipeline time: {total_time:.2f} sec")
    print(f"Total inference time: {total_infer_time:.2f} sec")
    print("DONE")

# ---------------- MULTI VIDEO ----------------
def run_on_folder(video_dir, model_path, output_dir):
    video_dir = Path(video_dir)

    video_files = sorted([
        f for f in video_dir.iterdir()
        if f.suffix.lower() in [".mp4", ".avi"]
    ])

    if len(video_files) == 0:
        print("No videos found.")
        return

    print(f"Found {len(video_files)} videos.\n")

    overall_start = time.time()

    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")

        try:
            run(video_path, model_path, output_dir)
        except Exception as e:
            print(f"Error processing {video_path.name}: {e}")
            continue

    total_time = time.time() - overall_start
    print(f"\nAll videos processed in {total_time:.2f} sec")

# ---------------- RUN ----------------
if __name__ == "__main__":
    VIDEO_DIR = BASE_DIR.parent / "model validation 2"
    MODEL_PATH = BASE_DIR / "model.pt"
    OUTPUT_DIR = BASE_DIR / "results_unsupervised"

    run_on_folder(VIDEO_DIR, MODEL_PATH, OUTPUT_DIR)