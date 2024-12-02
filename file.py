import os
import cv2
from openai import OpenAI
from deepface import DeepFace
import speech_recognition as sr
import threading
from gtts import gTTS
from playsound import playsound
import tempfile

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

if not client.api_key:
    raise ValueError("OpenAI API Key is not set. Make sure to export it with 'export OPENAI_API_KEY=your_key_here'.")

# Initialize shared variables for face attributes
last_face_attributes = None
face_scanning_active = False

def speak_text(text):
    """Convert text to speech using gTTS and play it."""
    try:
        # Generate speech
        tts = gTTS(text=text, lang='en', tld='co.uk')  # Use 'co.uk' for a British accent
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_file:
            tts.save(temp_file.name)
            playsound(temp_file.name)
    except Exception as e:
        print(f"Error during TTS: {e}")
        
def get_gpt_response(attributes, user_input):
    """
    Generate a response from GPT based on face attributes and user input.
    Args:
        attributes (dict): Detected facial attributes.
        user_input (str): User's spoken input.
    Returns:
        str: GPT response.
    """
    try:
        prompt = (
            f"The detected person has the following attributes: {attributes}. "
            f"They asked: '{user_input}'. Respond as the Magic Mirror from Shrek."
        )
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are the Magic Mirror from Shrek. Be witty, sarcastic, and playful in your responses."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error while generating response: {e}")
        return "I seem to have lost my magical connection. Try again later!"

def listen_for_phrase():
    """Listen for the activation phrase 'Mirror, mirror on the wall'."""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Listening for the activation phrase...")
    while True:
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio).lower()
            print(f"Heard: {command}")

            if "hello" in command:
                speak_text("Yes, I'm here. What do you desire?")
                return True
        except sr.UnknownValueError:
            print("Could not understand the audio. Listening again...")
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            break
    return False

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

def handle_conversation():
    """Handle the conversation loop after activation."""
    global face_scanning_active, last_face_attributes

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while True:
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
            user_input = recognizer.recognize_google(audio).lower()
            print(f"You said: {user_input}")

            if "goodbye" in user_input:
                speak_text("Farewell, mortal. Until we meet again!")
                face_scanning_active = False  # Stop face scanning
                break

            if last_face_attributes:
                # Generate and speak GPT response
                response = get_gpt_response(last_face_attributes, user_input)
                print(f"Mirror says: {response}")
                print(f"Last Face Attributes: {last_face_attributes}")
                speak_text(response)
            else:
                speak_text("I cannot see your face. Please step in front of the mirror.")
        except sr.UnknownValueError:
            print("Could not understand the audio. Listening again...")
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            break

if __name__ == "__main__":
    if listen_for_phrase():
        # Start face scanning in a separate thread
        face_scanning_thread = threading.Thread(target=scan_faces)
        face_scanning_thread.start()

        # Start handling conversation
        handle_conversation()

        # Wait for face scanning thread to finish
        face_scanning_thread.join()