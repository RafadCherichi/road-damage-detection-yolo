"""
Unified inference script: image / folder / video input,
supports both .pt and .onnx weights.
"""
import argparse
import time
from pathlib import Path
import cv2
from ultralytics import YOLO


def load_model(weights_path):
    # Ultralytics' YOLO() class auto-detects .pt vs .onnx and picks the
    # right backend — no branching needed here, which keeps this simple.
    return YOLO(weights_path)


def run_on_image(model, image_path, out_dir, conf_thres):
    results = model.predict(image_path, conf=conf_thres, verbose=False)[0]
    annotated = results.plot()
    out_path = Path(out_dir) / f"pred_{Path(image_path).name}"
    cv2.imwrite(str(out_path), annotated)
    return out_path


def run_on_folder(model, folder_path, out_dir, conf_thres):
    exts = (".jpg", ".jpeg", ".png")
    images = [f for f in Path(folder_path).iterdir() if f.suffix.lower() in exts]
    for img_path in images:
        run_on_image(model, str(img_path), out_dir, conf_thres)
    print(f"Processed {len(images)} images -> {out_dir}")


def run_on_video(model, video_path, out_dir, conf_thres):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(out_dir) / f"pred_{Path(video_path).name}"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_times = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()
        results = model.predict(frame, conf=conf_thres, verbose=False)[0]
        frame_times.append(time.time() - t0)
        writer.write(results.plot())

    cap.release()
    writer.release()
    if frame_times:
        avg_ms = (sum(frame_times) / len(frame_times)) * 1000
        print(f"Processed video -> {out_path}")
        print(f"Avg inference time: {avg_ms:.1f}ms/frame ({1000/avg_ms:.1f} FPS)")
    else:
        print("No frames processed.")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent 
    default_out = str(PROJECT_ROOT / "results" / "inference_samples")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help=".pt or .onnx path")
    parser.add_argument("--source", required=True, help="image, folder, or video path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", default=default_out)
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model = load_model(args.weights)
    src = Path(args.source)

    if src.is_dir():
        run_on_folder(model, src, args.out, args.conf)
    elif src.suffix.lower() in (".mp4", ".avi", ".mov"):
        run_on_video(model, str(src), args.out, args.conf)
    else:
        run_on_image(model, str(src), args.out, args.conf)