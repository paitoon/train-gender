import os
import glob
import cv2

def mark_faces(classNo, imageNameList, outDir, trainStream):
    for imageName in imageNameList:
        imagePath = os.path.join(outDir, imageName)
        trainStream.write(imagePath)
        trainStream.write('\n')

        image = cv2.imread(imagePath)
        height, width, channel = image.shape

        namePair = imageName.split('.')
        markedImageName = "{}.txt".format(namePair[0])
        markedImagePath = os.path.join(outDir, markedImageName)
        of = open(markedImagePath, 'w')
        of.write("{} {} {} 1.0 1.0\n".format(classNo, width/2, height/2))
        of.close()

def main():
    facePath = "faces"

    if not os.path.exists(facePath):
        os.mkdir(facePath)
    
    trainStream = open("faces/train.txt", "w")

    print("Retrieving male face images ...")
    males = [name for name in os.listdir("faces/male") if name.endswith(".jpg")]
    print("Retrieved {} faces !".format(len(males)))

    mark_faces(0, males, "faces/male", trainStream)

    print("Retrieving female face images ...")
    females = [name for name in os.listdir("faces/female") if name.endswith(".jpg")]
    print("Retrieved {} faces !".format(len(females)))

    mark_faces(1, females, "faces/female", trainStream)

    trainStream.close()

if __name__ == "__main__":
    main()
