import cv2
import sys

cascpath = 'haarcascade_frontalface_default.xml'
cascaded_face = cv2.CascadeClassifier(cascpath)
video_show = cv2.VideoCapture(0)

while True:
	ret , frame = video_show.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = cascaded_face.detectMultiScale(
			gray ,
			scaleFactor = 1.1,
			minNeighbors = 5 ,
			minSize = (5,5),
			flags = cv2.CASCADE_SCALE_IMAGE
		)

	for x , y , w , h in faces :
		cv2.rectangle(frame , (x,y) , (x+w , y+h) , (120,120,255) , 3)

	cv2.imshow('Video' , frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_show.release()
cv2.destroyAllWindows()