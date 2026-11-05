
import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam!")
    exit()

print("Starting Real-Time Emotion Detection...")
print("Press ESC to exit.")

frame_count = 0
face_results = []

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error reading webcam frame!")
        break

    frame_count += 1

    # Detect emotions every 5 frames
    if frame_count % 5 == 0:

        try:
            results = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                detector_backend='opencv',
                enforce_detection=False,
                silent=True
            )

            if isinstance(results, dict):
                results = [results]

            face_results = results

        except Exception as e:
            print("Detection error:", e)

    # Draw face rectangles and emotion labels
    for result in face_results:

        area = result.get("facial_area", {})

        x = area.get("x", 0)
        y = area.get("y", 0)
        w = area.get("w", 0)
        h = area.get("h", 0)

        if w <= 0 or h <= 0:
            continue

        emotion = result.get(
            "dominant_emotion", "Unknown"
        )

        confidence = result.get(
            "emotion", {}
        ).get(emotion, 0)

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display emotion and confidence
        label = f"{emotion.upper()} ({confidence:.1f}%)"

        cv2.putText(
            frame,
            label,
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "REAL-TIME EMOTION DETECTION",
        frame
    )

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
