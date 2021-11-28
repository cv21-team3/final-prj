from imutils import face_utils
import argparse
import imutils
import dlib
import cv2


# @param image: segmentor 를 통해 얻은 사람(들) region 외 mask 된 이미지
# @retval image: face detected image (if any)
def show_raw_detection(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    facial_coord=[]
    # Detect face in the grayscale image
    # @param gray scale image
    # @param upscaling factor
    rects = detector(gray, 3)

    if len(rects) != 0:
        for (i, rect) in enumerate(rects):

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            facial_coord.append((x,y,w,h))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(image, "Face #{}".format(i), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        return image, facial_coord

    else:
       # print("No face detected.")
        return image, facial_coord


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    args = vars(ap.parse_args())

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    image = cv2.imread(args["image"])
    image = imutils.resize(image, width=500)

    complete_landmarks_detected_img,_ = show_raw_detection(image, detector, predictor)

    if complete_landmarks_detected_img is not None:
        cv2.imwrite("Output.jpeg", complete_landmarks_detected_img)
        if cv2.waitKey(0) == 27:    # Press ESC to exit
            cv2.destroyAllWindows()
    else:
        print("Frontal face not detected. Search for non-frontal or occluded face.")

if __name__ == '__main__':
    main()
