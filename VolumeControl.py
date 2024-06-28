import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# setting the cam parameters(width & height)
wCam, hCam = 700, 500

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

# print(f"Requested Width : {wCam}, Requested Height : {hCam}", flush=True)
# print(f"Actual Width : {actual_width}, Actual Height : {actual_height}", flush=True)

time.sleep(2)
pTime = 0

detector = htm.handDetector(detectionCon = 0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
# print(volume.GetVolumeRange())
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)
    if len(lmList) != 0:
        #print(lmList[4],lmList[8])    # landmarks no 4 and 8 correspond to the tip of thumb and the adjacent finger respectively(both are used for adjusting volume level)
        x1, y1 = lmList[4][1], lmList[4][2]    # example : [4, 340, 645] represents id, x1, y1
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 12, (0, 150, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 12, (0, 150, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 205, 0), 3)
        cv2.circle(img, (cx, cy), 10, (0, 150, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        #print(length)

        # Hand range : 50 - 300 , Vol range : -65 to 0
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])
        print(int(length),vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 6, (255, 0, 255), cv2.FILLED)

        cv2.rectangle(img, (50, 150), (85, 450), (255,0,0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 450), (255,0,0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)}%', (40, 135), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS : {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)