import cv2
import torch
from ultralytics import YOLO
import time

# Set device and optimize for performance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True  # Optimize for performance on GPU

def main():
    # Load YOLOv9 model with half precision (faster inference)
    model = YOLO('yolov9c.pt')
    model.to(device)
    model.fuse()  # Fuse Conv2d + BatchNorm layers for faster inference
    
    # Initialize video capture with optimized settings
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for faster camera access on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # Lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)             # Set FPS to 30
    
    if not cap.isOpened():
        print("Error: Could not open video capture device")
        return
    
    # Warmup the model
    print("Warming up the model...")
    _ = model(torch.zeros(1, 3, 640, 480).to(device)) if torch.cuda.is_available() else None
    
    # Initialize FPS counter
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (640, 480))
        
        # Run inference with half precision if on GPU
        with torch.no_grad():
            results = model(frame_resized, verbose=False, imgsz=640)  # Disable verbose output for speed
        
        # Process the results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Move to CPU before conversion
                confidence = box.conf[0].item()
                
                # Only process detections above a confidence threshold
                if confidence > 0.5:  # Increased threshold for faster processing
                    label_id = int(box.cls[0].item())
                    class_label = model.names[label_id]
                    label_text = f"{class_label}: {confidence:.2f}"
                    
                    # Scale coordinates back to original frame size
                    h, w = frame.shape[:2]
                    x1, y1, x2, y2 = int(x1 * w/640), int(y1 * h/480), int(x2 * w/640), int(y2 * h/480)
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame with bounding boxes and labels
        cv2.imshow('Real-Time Object Detection (Press Q to quit)', frame)
        
        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
