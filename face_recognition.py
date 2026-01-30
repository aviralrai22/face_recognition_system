import cv2
import numpy as np
import os 

# --- SETUP ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml') # Load the trained model

# Use the internal path to avoid "file not found" errors
cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# --- NAMES SETUP ---
# The ID matches the number you typed during data collection (1, 2, 3...)
# Note: The list starts at 0, so 'None' is for ID 0 (which usually isn't used).
# ID 1 = Your Name, ID 2 = Friend, etc.
names = ['None', 'Aviral Rai', 'tanay', 'kuldeep'] 

# Initialize Webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640) # Width
cam.set(4, 480) # Height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

print("\n [INFO] Starting Camera... Press 'ESC' to exit.")

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale (algorithm requirement)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        
        # Recognize the face using the 'trainer.yml'
        # id = the ID number (1, 2, 3...)
        # confidence = how "different" the face is (0 is perfect match)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # If confidence is less than 100, it's a match
        # (Lower confidence number = Better match in LBPH)
        if (confidence < 100):
            try:
                id_name = names[id]
            except IndexError:
                id_name = "Unknown ID"
            
            # Calculate a % match (just for display)
            confidence_text = "  {0}%".format(round(100 - confidence))
        else:
            id_name = "unknown"
            confidence_text = "  {0}%".format(round(100 - confidence))
        
        # Display the name
        cv2.putText(img, str(id_name), (x+5,y-5), font, 1, (255,255,255), 2)
        # Display the confidence %
        cv2.putText(img, str(confidence_text), (x+5,y+h-5), font, 1, (255,255,0), 1)  

    cv2.imshow('Face Recognition', img) 

    # Press 'ESC' to quit
    k = cv2.waitKey(10) & 0xff 
    if k == 27: 
        break

print("\n [INFO] Exiting Program.")
cam.release()
cv2.destroyAllWindows()