import os
import glob
import dlib
import cv2

face_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
mode = "hog"
detector = dlib.get_frontal_face_detector()
if mode == "cnn":
    detector = dlib.cnn_face_detection_model_v1(face_model)

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

def mark_faces(classNo, imagePathList, outDir, trainStream):
    for imagePath in imagePathList:
        dirName = os.path.dirname(imagePath)
        baseName = os.path.basename(imagePath)
        txtName = "{}.txt".format(baseName.split(".")[0])
        print("Process...{}".format(baseName))

        image = cv2.imread(imagePath)
        imageHeight, imageWidth, _ = image.shape
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faceLocations = detector(image, 1)
        for faceLocation in faceLocations:
            trainStream.write(imagePath)
            trainStream.write('\n')

            css = _rect_to_css(faceLocation)
            boundRectangle = _trim_css_to_bounds(css, rgbImage.shape)
            top = int(boundRectangle[0])
            left = int(boundRectangle[3])
            right = int(boundRectangle[1])
            bottom = int(boundRectangle[2])

            x_center = ((left + right) / 2) / imageWidth
            y_center = ((top + bottom) / 2) / imageHeight
            width = (right - left) / imageWidth
            height = (bottom - top) / imageHeight
            markedImagePath = os.path.join(outDir, txtName)
            of = open(markedImagePath, 'w')
            of.write("{} {} {} {} {}\n".format(classNo, x_center, y_center, width, height))
            of.close()

def main():
    trainStream = open("images/train.txt", "w")
    
    print("Retrieving males images ...")
    imagePath = "images/males"
    males = glob.glob(os.path.join(imagePath, "*.jpg"))
    print("Retrieved {} faces !".format(len(males)))

    mark_faces(0, males, imagePath, trainStream)

    print("Retrieving females images ...")
    imagePath = "images/females"
    females = glob.glob(os.path.join(imagePath, "*.jpg"))
    print("Retrieved {} faces !".format(len(females)))

    mark_faces(1, females, imagePath, trainStream)

if __name__ == "__main__":
    main()
