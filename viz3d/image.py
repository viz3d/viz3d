import cv2
import numpy as np
import config
import time

if __name__ == "__main__":
    cfg = config.read_config()

    left = cv2.VideoCapture(cfg["cameras"]["left"])
    right = cv2.VideoCapture(cfg["cameras"]["right"])

    left.set(cv2.CAP_PROP_FPS, cfg["cameras"]["fps"])
    right.set(cv2.CAP_PROP_FPS, cfg["cameras"]["fps"])

    while True:
        ret, frameLeft = left.read()
        if not ret:
            print("Cannot load left camera!")
            break
        ret, frameRight = right.read()
        if not ret:
            print("Cannot load right camera!")
            break

        grayLeft = cv2.cvtColor(frameLeft, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(frameRight, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frameLeft', grayLeft)
        cv2.imshow('frameRight', grayRight)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            epoch = int(time.time())
            cv2.imwrite("../captures/%i-left.png" % epoch, grayLeft)
            cv2.imwrite("../captures/%i-right.png" % epoch, grayRight)

    left.release()
    right.release()
    cv2.destroyAllWindows()
