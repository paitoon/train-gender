import pickle
import dlib
import face_recognition_models
import numpy as np
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

def adjust_gamma(input_image, gamma=1.0):
    table = np.array([((iteration / 255.0) ** (1.0 / gamma)) * 255
                      for iteration in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(input_image, table)


def read_image(path, gamma=0.75):
    output = cv2.imread(path)
    return adjust_gamma(output, gamma=gamma)


def face_vectors(input_image):
    faces = detector(input_image, 1)
    if not faces:
        return None

    face_descriptors = []
    for face in faces:
        shape = predictor(input_image, face)
        face_descriptor = face_model.compute_face_descriptor(input_image, shape)
        face_descriptors.append(face_descriptor)
    return face_descriptors

def is_male(p, thresh=0.5):
	return p > thresh

def is_female(p, thresh=-0.5):
	return p < thresh

cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

classifier = pickle.load(open('gender_model.pickle', 'rb'))
# face_descriptor from compute_face_descriptor function
image = read_image("test/people.jpg")
faceLocations = cnn_face_detector(image, 1)
face_descriptors = face_vectors(image)
for face_descriptor in face_descriptors:
    prediction = classifier(face_descriptor)
print(prediction)
