import glob
import dlib
import face_recognition_models
import cv2
import pickle
import random
import numpy as np
import sys


detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

def adjust_gamma(input_image, gamma=1.0):
    table = np.array([((iteration / 255.0) ** (1.0 / gamma)) * 255
                      for iteration in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(input_image, table)


def read_image(path, gamma=0.75):
    output = cv2.imread(path)
    return adjust_gamma(output, gamma=gamma)


def face_vector(input_image):
    faces = detector(input_image, 1)
    if not faces:
        return None

    f = faces[0]
    shape = predictor(input_image, f.rect)
    face_descriptor = face_model.compute_face_descriptor(input_image, shape)
    return face_descriptor


def _rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order
    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])

def _trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

if sys.argv[1] == 'train':
    max_size = 1000
    male_label = +1
    female_label = -1

    print("Retrieving males images ...")
    males = glob.glob("images/males/*.jpg")
    print("Retrieved {} faces !".format(len(males)))

    print("Retrieving females images ...")
    females = glob.glob("images/females/*.jpg")
    print("Retrieved {} faces !".format(len(females)))

    females = females[:max_size]
    males = males[:max_size]

    vectors = dlib.vectors()
    labels = dlib.array()

    print("Reading males images ...")
    for i, male in enumerate(males):
        print("Reading {} of {}\r".format(i, len(males)))
        face_vectors = face_vector(read_image(male))
        if face_vectors is None:
            continue
        vectors.append(dlib.vector(face_vectors))
        labels.append(male_label)

    print("Reading females images ...")
    for i, female in enumerate(females):
        print("Reading {} of {}\r".format(i, len(females)))
        face_vectors = face_vector(read_image(female))
        if face_vectors is None:
            continue
        vectors.append(dlib.vector(face_vectors))
        labels.append(female_label)

    svm = dlib.svm_c_trainer_linear()
    #svm = dlib.svm_c_trainer_radial_basis()
    svm.set_c(10)
    classifier = svm.train(vectors, labels)

    print("Prediction for male sample:  {}".format(classifier(vectors[random.randrange(0, max_size)])))
    print("Prediction for female sample: {}".format(classifier(vectors[max_size + random.randrange(0, max_size)])))

    with open('gender_model.pickle', 'wb') as handle:
        pickle.dump(classifier, handle)
else:
    print("Test sample image...")

    classifier = pickle.load(open('gender_model.pickle', 'rb'))
    # face_descriptor from compute_face_descriptor function
    #image = read_image("test/people.jpg")
    image = cv2.imread("test/people1.jpg")
    rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #faceLocations = cnn_face_detector(image, 1)
    faceLocations = detector(image, 1)
    predictions = []
    for faceLocation in faceLocations:
        css = _rect_to_css(faceLocation)
        boundRectangle = _trim_css_to_bounds(css, rgbImage.shape)
        shape = predictor(image, faceLocation)
        face_descriptor = face_model.compute_face_descriptor(image, shape)
        score = classifier(face_descriptor)
        if score > 0:
            prediction = { 'location': boundRectangle, 'gender': 'male', 'confidence': score }
        else:
            prediction = { 'location': boundRectangle, 'gender': 'female', 'confidence': score * -1 }
        predictions.append(prediction)
        print(prediction)

        location = prediction['location']
        top = int(location[0])
        right = int(location[1])
        bottom = int(location[2])
        left = int(location[3])

        # draw the predicted face name on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, prediction['gender'], (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite('test/prdict.jpg', image)
    