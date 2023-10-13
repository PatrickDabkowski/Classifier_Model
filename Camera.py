import Classifier_Model
import cv2
import os
import time
import argparse

# Video settings
cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 450)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 330)


def classify(img, model):
    img_copy = img.copy()
    img_copy = cv2.resize(img_copy, dsize=(28, 28), interpolation=cv2.INTER_LINEAR)
    img_copy = 255 - img_copy
    result = model.predict(img_copy.reshape((1, 28, 28)))
    return result

def camera_inference(model):
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

        result = classify(frame, model)
        result = result.argmax(axis=-1)
        time.sleep(0.1)
        print('number ', result)
        cv2.imshow('Video Digit Classifier', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera model inference')
    parser.add_argument('model_path', type=str, default="Model.CNN", help="path to your trained model")
    parser.add_argument('target_path', type=str, default="Model.CNN", help="path to save your model")

    # Model
    model = Classifier_Model.get_model(parser.model_path, parser.target_path)
    
    camera_inference(model)
