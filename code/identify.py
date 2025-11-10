from ultralytics import YOLO
import cv2
import os

def detect_persons(
    image_path: str,
    model_path: str = "yolov8n.pt",   # can also use trained .pt/.onnx
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: str | None = None         
):
    """
    读取一张图片，使用 YOLO 模型检测人的目标，返回检测结果列表。
    返回的每一项为:
    read a single image, use yolo to detect human and return results as listed below:
    {
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_xywh": [x, y, w, h],
        "conf": float,
        "class_id": int,
        "class_name": str,
        "image_size": (H, W)
    }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # image size read
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    H, W = img.shape[:2]

    # 加载模型 load the model 
    model = YOLO(model_path)

    # 只检测“人”（COCO 的 class 0） only detects human for now
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        classes=[0],
        device=device,
        verbose=False
    )

    r = results[0]
    out = []
    names = r.names  # 类别名字典

    if r.boxes is None:
        return out

    for b in r.boxes:
        # xyxy坐标 coradinates 
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        # 转成 xywh（左上角 + 宽高）
        xywh = [x1, y1, x2 - x1, y2 - y1]
        conf = float(b.conf[0])
        cls_id = int(b.cls[0])
        out.append({
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "bbox_xywh": [float(v) for v in xywh],
            "conf": conf,
            "class_id": cls_id,
            "class_name": names.get(cls_id, str(cls_id)),
            "image_size": (int(H), int(W)),
        })
    return out

def save_annotated_image(image_path: str, detections: list, save_path: str):
    """
    可选：把检测结果画在图上并保存。
    optional : save the results and plot on the graph.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox_xyxy"])
        label = f'{det["class_name"]} {det["conf"]:.2f}'
        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 标注文字
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imwrite(save_path, img)

# ================== sample usage 使用示例 ==================
if __name__ == "__main__":
    img_path = "../images/singleperson.jpg"              # 你的图片路径
    model_path = "yolov8n.pt"          # 或者自定义模型，如 runs/detect/train/weights/best.pt
    detections = detect_persons(img_path, model_path, conf_threshold=0.3, iou_threshold=0.5, device=None)
    print(detections)

    # 如需可视化：
    if detections:
        save_annotated_image(img_path, detections, "annotated.jpg")
        print("Annotated image saved to annotated.jpg")

