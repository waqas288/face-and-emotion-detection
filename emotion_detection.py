import cv2
from deepface import DeepFace

# List of available algorithms in DeepFace
available_algorithms = ['VGG-Face', 'GoogleFaceNet', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib']

# Function for live facial detection and emotion recognition
def live_facial_detection_and_emotion(algorithm):
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Detect emotions
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            emotion = DeepFace.analyze(face, actions=['emotion'])

            # Get the dominant emotion
            emotion_label = max(emotion[0]['emotion'].items(), key=lambda x: x[1])[0]

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display the emotion label
            cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame with rectangles around faces and emotion labels
        cv2.imshow('Live Facial Detection with Emotion', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Main code to select algorithm and run live facial detection with emotion recognition
if __name__ == "__main__":
    print("Available algorithms in DeepFace:")
    for idx, alg in enumerate(available_algorithms):
        print(f"{idx+1}. {alg}")

    try:
        selected_idx = int(input("Enter the number of the algorithm you want to use (1-7): ")) - 1
        selected_algorithm = available_algorithms[selected_idx]
        print(f"Selected algorithm: {selected_algorithm}")

        live_facial_detection_and_emotion(selected_algorithm)

    except ValueError:
        print("Invalid input. Please enter a valid number.")
