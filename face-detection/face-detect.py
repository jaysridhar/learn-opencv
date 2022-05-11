#!/usr/bin/python

import cv2, time, argparse
from ptpython.repl import embed

parser = argparse.ArgumentParser(description='OpenCV Face Recognition')
parser.add_argument('-interactive', action='store_true', help='Open a console terminal for evaluation')
parser.add_argument('input', help='Input video file to process')
args = parser.parse_args()
print(args)

import mediapipe as mp

cam = cv2.VideoCapture(args.input)
pTime = time.time()

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
fd = mpFaceDetection.FaceDetection()

while True:
    rv, img = cam.read()
    if not rv: break
    hT, wT, cT = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = fd.process(imgRGB)
    if args.interactive: embed(globals(), locals())
    elif results.detections:
        for i, detection in enumerate(results.detections):
            score = int(100*detection.score[0]) if len(detection.score) > 0 else 0
            bbox = detection.location_data.relative_bounding_box
            w, h = int(bbox.width*wT), int(bbox.height*hT)
            x, y = int(bbox.xmin*wT), int(bbox.ymin*hT)
            mpDraw.draw_detection(img, detection)
            cv2.putText(img, f'[{i+1}] {score}%',
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 255),
                        2)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow('Image', img)
    if args.interactive:
        cv2.waitKey(1)
        break
    else:
        if cv2.waitKey(1) == ord('q'): break
