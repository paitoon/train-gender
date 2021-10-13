import glob
import dlib
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

gender_datasets = 'gender_datasets.pickle'
gender_model = 'gender_model.pickle'

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

def adjust_gamma(input_image, gamma=0.75):
    table = np.array([((iteration / 255.0) ** (1.0 / gamma)) * 255
                      for iteration in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(input_image, table)

def build_class(classNo, imagePathList, data, target):
    for imagePath in imagePathList:
        #print("Process image : ", imagePath)
        rawImage = cv2.imread(imagePath)
        image = adjust_gamma(rawImage, 0.9)
        #image = cv2.cvtColor(rawImage, cv2.COLOR_BGR2RGB)
        faceLocations = detector(image, 1)
        if not faceLocations:
            print("Can not detect face location in : ", imagePath)
            continue
        for faceLocation in faceLocations:
            shape = predictor(image, faceLocation.rect)
            face_descriptor = face_model.compute_face_descriptor(image, shape)
            face_vectors = dlib.vector(face_descriptor)
            data.append(list(face_vectors))
            target.append(classNo)
    return data, target

def build_data():
    print("Retrieving males images ...")
    males = glob.glob("images/males/*.jpg")
    print("Retrieved {} faces !".format(len(males)))

    print("Retrieving females images ...")
    females = glob.glob("images/females/*.jpg")
    print("Retrieved {} faces !".format(len(females)))

    data = []
    target = []

    data, target = build_class(0, males, data, target)
    data, target = build_class(1, females, data, target)

    datasets = { "data": data, "target": target }
    with open(gender_datasets, 'wb') as handle:
        pickle.dump(datasets, handle)
    return datasets

def load_data():
    datasets = pickle.load(open(gender_datasets, 'rb'))
    print("Load {} data sets, {} targets...".format(len(datasets["data"]), len(datasets["target"])))
    return datasets

def build_model(datasets):
    if datasets == None:
        print("No datasets...")
        return
    
    print("Split 70/30 of {} data sets, {} targets...".format(len(datasets["data"]), len(datasets["target"])))
    #x_train, x_test, y_train, y_test = train_test_split(datasets["data"], datasets["target"], test_size=0.3,random_state=109)
    x_train, x_test, y_train, y_test = train_test_split(datasets["data"], datasets["target"], test_size=0.3)

    classifier = svm.SVC(kernel='linear', probability=True)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:",metrics.recall_score(y_test, y_pred))

    pickle.dump(classifier, open(gender_model, 'wb'))
    return classifier

def load_model():
    model = pickle.load(open(gender_model, 'rb'))
    print("Load model {} ...".format(gender_model))
    return model

def predict(model):
    imagePath = input("Enter image file : ")
    rawImage = cv2.imread(imagePath)
    image = adjust_gamma(rawImage, 0.9)
    #image = cv2.cvtColor(rawImage, cv2.COLOR_BGR2RGB)
    faceLocations = detector(image, 1)
    if not faceLocations:
        print("Can not detect face location in : ", imagePath)
        return
    
    data = []
    rectangles = []
    for faceLocation in faceLocations:
        shape = predictor(image, faceLocation.rect)
        face_descriptor = face_model.compute_face_descriptor(image, shape)
        face_vectors = dlib.vector(face_descriptor)
        data.append(list(face_vectors))

        css = _rect_to_css(faceLocation.rect)
        boundRectangle = _trim_css_to_bounds(css, image.shape)
        rectangles.append(boundRectangle)

    targets = model.predict(data)
    probs = model.predict_proba(data)
    for i, target in enumerate(targets):
        name = "female"
        if target == 0:
            name = "male"
        prob = probs[i][target]
        # draw the predicted face name on the image
        boundRectangle = rectangles[i]
        top = int(boundRectangle[0])
        left = int(boundRectangle[3])
        right = int(boundRectangle[1])
        bottom = int(boundRectangle[2])
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, "{}:{}".format(name, round(prob, 2)), (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite("predicted_image.jpg", image)
    print("Result : ", targets)

def main():
    running = True
    datasets = None
    model = None
    while running:
        print("1. Create data set")
        print("2. Load data set")
        print("3. Build SVM Model")
        print("4. Load SVM Model")
        print("5. Prediction")
        print("E. Exit")

        cmd = input("Enter command : ")
        if cmd == "1":
            datasets = build_data()
        elif cmd == "2":
            datasets = load_data()
        elif cmd == "3":
            model = build_model(datasets)
        elif cmd == "4":
            model = load_model()
        elif cmd == "5":
            predict(model)
        elif cmd.lower() == "e":
            running = False

if __name__ == "__main__":
    main()
