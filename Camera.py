import Classifier_Model
import cv2
import os
import time

# Video settings
cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 450)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 330)

#Model
model = Classifier_Model.model

def classify(img):
    img_copy = img.copy()
    img_copy = cv2.resize(img_copy, dsize=(28, 28), interpolation=cv2.INTER_LINEAR)
    img_copy = 255 - img_copy
    result = model.predict(img_copy.reshape((1, 28, 28)))
    return result

while cap.isOpened():
    ret, frame = cap.read(0)
    if not cap.isOpened():
        os.popen('open -a ScreenSaverEngine')
        raise Exception("Could not open video device")
    else:
        print('Success Video Capturing with: ')
    # Fliping mirror effect
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.flip(frame, 1)
    thresh = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    frame = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    result = classify(frame)
    result = result.argmax(axis=-1)
    time.sleep(0.11)
    print('number ', result)
    cv2.imshow('Video Digit Classifier', frame)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()