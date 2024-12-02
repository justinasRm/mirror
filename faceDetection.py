import cv2
from deepface import DeepFace

def scan_faces():
    """Continuously scan for faces and update the last detected attributes."""
    global last_face_attributes, face_scanning_active
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_scanning_active = True

    # Counter for consecutive detections
    consecutive_face_count = 0
    required_consecutive_detections = 5  # Number of consecutive detections to confirm a face

    while face_scanning_active:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to access webcam.")
            continue

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(100, 100))

        if len(faces) > 0:
            consecutive_face_count += 1
            print(f"Face detected. Consecutive count: {consecutive_face_count}")
            if consecutive_face_count >= required_consecutive_detections:
                # Reset counter to avoid re-processing the same face
                consecutive_face_count = 0

                for (x, y, w, h) in faces:
                    # Crop the face from the frame
                    face_img = frame[y:y+h, x:x+w]

                    try:
                        # Analyze the face using DeepFace
                        analysis = DeepFace.analyze(face_img, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)

                        # Handle both list and dict results
                        if isinstance(analysis, list):
                            analysis = analysis[0]  # Take the first result if it's a list

                        # Update last detected attributes
                        last_face_attributes = {
                            "Age": analysis.get('age', 'Unknown'),
                            "Gender": analysis.get('gender', 'Unknown'),
                            "Emotion": max(analysis.get('emotion', {}).items(), key=lambda x: x[1])[0] if 'emotion' in analysis else "Unknown",
                            "Dominant Race": analysis.get('dominant_race', 'Unknown')
                        }
                        print(f"Updated Attributes: {last_face_attributes}")

                    except Exception as e:
                        print(f"Error analyzing face: {e}")
        else:
            # Reset the counter if no face is detected
            consecutive_face_count = 0
            print("No face detected.")

    cap.release()
    cv2.destroyAllWindows()


scan_faces()
