import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        # Ignore system files or folders
        if os.path.isdir(imagePath) or not imagePath.endswith(('.jpg', '.jpeg', '.png')):
            continue

        # 1. Load the image and convert to Gray
        PIL_img = Image.open(imagePath).convert('L') 
        img_numpy = np.array(PIL_img,'uint8')

        # 2. Get the ID
        try:
            id = int(os.path.split(imagePath)[-1].split(".")[1])
        except:
            print(f"Skipping file with bad name: {imagePath}")
            continue

        # 3. DIRECTLY use the image (Don't run face detector again)
        # Since we already cropped it in Step 1, we can just use the whole image
        faceSamples.append(img_numpy)
        ids.append(id)
        
        # Optional: Print to verify it's working
        # print(f"Loaded ID: {id} from {imagePath}")

    return faceSamples, ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') 

print(f"\n [INFO] {len(np.unique(ids))} faces trained. Exiting Program")