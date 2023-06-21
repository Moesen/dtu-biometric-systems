from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument(
#     "-p", "-shape-predictor", required=True, help="path to facial landmark predictor"
# )
# ap.add_argument("-i", "-image", required=True, help="path to input image")
# args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()  # type: ignore
