"""
Per-detection Grad-CAM for YOLOv8 (Ultralytics).
Fixes: sky-dominant EigenCAM, dead-gradient/global-max Grad-CAM,
and single-layer multi-scale mismatch.

For each detected box, backprops from that box's own class logit
through the specific P3/P4/P5 scale that produced it.
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Layer indices confirmed from this project's actual model summary:
# 15 -> P3/stride-8  (small objects, e.g. thin cracks)
# 18 -> P4/stride-16 (medium objects)
# 21 -> P5/stride-32 (large objects, e.g. alligator patches)
SCALE_LAYERS = {8: 15, 16: 18, 32: 21}
STRIDES = [8, 16, 32]


class DetectionGradCAM:
    def __init__(self, weights_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.yolo = YOLO(weights_path)
        self.model = self.yolo.model.to(device).eval()
        self.device = device
        self.class_names = self.yolo.names

        self._activations = {}
        self._gradients = {}
        for stride, layer_idx in SCALE_LAYERS.items():
            layer = self.model.model[layer_idx]
            layer.register_forward_hook(self._save_activation(stride))
            layer.register_full_backward_hook(self._save_gradient(stride))

    def _save_activation(self, stride):
        def hook(module, inp, out):
            self._activations[stride] = out
        return hook

    def _save_gradient(self, stride):
        def hook(module, grad_in, grad_out):
            self._gradients[stride] = grad_out[0]
        return hook

    def _infer_stride_for_box(self, box_xyxy, img_shape):
        """Heuristic: box area determines which scale most likely produced it.
        Larger boxes -> coarser stride (P5). Matches YOLO's own scale assignment logic."""
        x1, y1, x2, y2 = box_xyxy
        area = (x2 - x1) * (y2 - y1)
        img_area = img_shape[0] * img_shape[1]
        ratio = area / img_area
        if ratio < 0.02:
            return 8
        elif ratio < 0.10:
            return 16
        else:
            return 32

    def generate(self, image_path, conf_thres=0.25, imgsz=640):
        orig = cv2.imread(image_path)
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        h0, w0 = orig.shape[:2]

        results = self.yolo.predict(image_path, conf=conf_thres, imgsz=imgsz, verbose=False)[0]
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            return orig_rgb, np.zeros((h0, w0), dtype=np.float32), results

        img_tensor = torch.from_numpy(results.orig_img).to(self.device)  # placeholder replaced below
        # Rebuild the exact preprocessed tensor Ultralytics used (letterboxed, normalized)
        from ultralytics.data.augment import LetterBox
        lb = LetterBox(new_shape=(imgsz, imgsz))
        img_lb = lb(image=orig_rgb)
        img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device).requires_grad_(True)

        full_cam = np.zeros((imgsz, imgsz), dtype=np.float32)

        for box, cls_id, conf in zip(boxes.xyxy.cpu().numpy(),
                                      boxes.cls.cpu().numpy().astype(int),
                                      boxes.conf.cpu().numpy()):
            self.model.zero_grad()
            self._activations.clear()
            self._gradients.clear()

            # FIX: Clear YOLO's inference-mode anchor cache so PyTorch can safely track gradients
            if hasattr(self.model.model[-1], 'shape'):
                self.model.model[-1].shape = None

            # Force explicit gradient tracking for the forward pass
            with torch.enable_grad():
                raw_out = self.model(img_tensor)[0] if isinstance(self.model(img_tensor), (list, tuple)) else self.model(img_tensor)
            # raw_out shape: [1, 4+num_classes, num_anchors]
            num_classes = len(self.class_names)
            box_wh_scale = (box[2] - box[0]) * (box[3] - box[1])
            stride = self._infer_stride_for_box(box, (h0, w0))
            grid_size = imgsz // stride

            # Map box center -> grid cell at this stride, then -> flat anchor index
            cx = (box[0] + box[2]) / 2 * (imgsz / w0)
            cy = (box[1] + box[3]) / 2 * (imgsz / h0)
            gx, gy = int(cx // stride), int(cy // stride)
            gx, gy = min(gx, grid_size - 1), min(gy, grid_size - 1)

            offset = sum((imgsz // s) ** 2 for s in STRIDES if s < stride)
            anchor_idx = offset + gy * grid_size + gx

            class_logit = raw_out[0, 4 + cls_id, anchor_idx]
            class_logit.backward(retain_graph=True)

            act = self._activations[stride][0]      # [C, Hs, Ws]
            grad = self._gradients[stride][0]        # [C, Hs, Ws]
            weights = grad.mean(dim=(1, 2))           # global-avg-pool gradients per channel
            cam = torch.relu((weights[:, None, None] * act).sum(dim=0))
            cam = cam.detach().cpu().numpy()
            cam = cv2.resize(cam, (imgsz, imgsz))
            if cam.max() > 0:
                cam = cam / cam.max()

            full_cam = np.maximum(full_cam, cam)  # accumulate per-box CAMs

        full_cam = cv2.resize(full_cam, (w0, h0))
        return orig_rgb, full_cam, results

    def save_comparison(self, image_path, out_path, conf_thres=0.25):
        orig_rgb, cam, results = self.generate(image_path, conf_thres=conf_thres)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(orig_rgb, 0.55, heatmap, 0.45, 0)

        detection_img = results.plot()[:, :, ::-1]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(detection_img)
        axes[0].set_title("YOLO Detection")
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title("Per-Detection Grad-CAM (correct scale, class-conditioned)")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_folder(weights_path, image_folder, output_folder, conf_thres=0.25):
    os.makedirs(output_folder, exist_ok=True)
    cam_gen = DetectionGradCAM(weights_path)
    for fname in sorted(os.listdir(image_folder)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(image_folder, fname)
        out_path = os.path.join(output_folder, f"gradcam_{os.path.splitext(fname)[0]}.jpg")
        try:
            cam_gen.save_comparison(img_path, out_path, conf_thres=conf_thres)
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Failed on {fname}: {e}")


if __name__ == "__main__":
    run_folder(
        weights_path=r"C:\Projects\road-damage-detection-yolo\results\training_runs\yolov8n_kaggle_run\weights\best.pt",
        image_folder=r"C:\Projects\road-damage-detection-yolo\data\raw\valid\images",
        output_folder=r"C:\Projects\road-damage-detection-yolo\results\gradcam_outputs",
        conf_thres=0.25,
    )