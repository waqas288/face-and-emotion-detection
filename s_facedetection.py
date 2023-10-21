import cv2
from deepface import DeepFace

# List of available algorithms in DeepFace
available_algorithms = ['VGG-Face', 'GoogleFaceNet', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib']

# Function for live facial detection
def live_facial_detection(algorithm):
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

        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the frame with rectangles around faces
        cv2.imshow('Live Facial Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Main code to select algorithm and run live facial detection
if __name__ == "__main__":
    print("Available algorithms in DeepFace:")
    for idx, alg in enumerate(available_algorithms):
        print(f"{idx+1}. {alg}")

    try:
        selected_idx = int(input("Enter the number of the algorithm you want to use (1-7): ")) - 1
        selected_algorithm = available_algorithms[selected_idx]
        print(f"Selected algorithm: {selected_algorithm}")

        live_facial_detection(selected_algorithm)

    except ValueError:
        print("Invalid input. Please enter a valid number.")
