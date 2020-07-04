import keras
import tensorflow
import numpy as np
import cv2
from keras.models import load_model
from time import sleep 
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import pathlib 
from pathlib import Path

root = Path(".")
path_to_model = root / "emotion_rec" / "Emotion_little_vgg.h5"
path_to_cascade = root / "emotion_rec" / "haarcascade_frontalface_defaults.xml"

face_detector = cv2.CascadeClassifier(str(path_to_cascade))
classifier = load_model(path_to_model)

emotion_labels = ("Angry", "Happy", "Neutral", "Sad", "Surprise")

def emotion_detect(image):
	image = cv2.imread(image) 
	emotions = []
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_detector.detectMultiScale(image, 1.3, 5)

	for (x, y, w, h) in faces:
		
		cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
		roi = image[y:y+h,x:x+w]
		roi = cv2.resize(roi, (48, 48), interpolation = cv2.INTER_AREA)
		
		if np.sum([roi]) != 0:
		
			roi = roi.astype("float") / 255.0
			roi = img_to_array(roi)
			roi = np.expand_dims(roi, axis = 0)

			predict = classifier.predict(roi)[0]
			predict = predict.argmax()
			emotion = emotion_labels[predict]
			emotions.append(emotion)

		else:
			continue

	return emotions   
