import cv2
import supervision as sv
from rfdetr import RFDETRBase, RFDETRSegPreview
from rfdetr.util.coco_classes import COCO_CLASSES


def find_available_cameras(max_cameras=10):
    """Find all available cameras and return their indices."""
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras


def select_camera():
    """Let user select which camera to use."""
    cameras = find_available_cameras()
    
    if not cameras:
        print("No cameras found!")
        return None
    
    print(f"\nAvailable cameras: {cameras}")
    
    if len(cameras) == 1:
        camera_idx = cameras[0]
        print(f"Using camera {camera_idx}")
        return camera_idx
    
    # Try camera 1 first (usually built-in Mac camera), then 0
    default_camera = 1 if 1 in cameras else cameras[0]
    
    user_input = input(f"Enter camera index to use (default={default_camera}, or press Enter): ").strip()
    
    if user_input == "":
        return default_camera
    
    try:
        camera_idx = int(user_input)
        if camera_idx in cameras:
            return camera_idx
        else:
            print(f"Invalid camera index. Using default: {default_camera}")
            return default_camera
    except ValueError:
        print(f"Invalid input. Using default: {default_camera}")
        return default_camera


# model = RFDETRBase()
model = RFDETRSegPreview()

camera_idx = select_camera()
if camera_idx is None:
    print("No camera available. Exiting.")
    exit(1)

cap = cv2.VideoCapture(camera_idx)

if not cap.isOpened():
    print(f"Failed to open camera {camera_idx}")
    exit(1)

print("\nStarting detection... Press 'q' to quit")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read frame")
        break

    # Convert BGR to RGB with copy to avoid negative stride issues
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = model.predict(rgb_frame, threshold=0.5)

    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = sv.MaskAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

    cv2.imshow("Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()