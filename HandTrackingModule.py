import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands =self.mpHands.Hands(static_image_mode = self.mode,
                                       max_num_hands = self.maxHands,
                                       min_detection_confidence = self.detectionCon,
                                       min_tracking_confidence = self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:    # get the index no and landmark(coordinates) of hands
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True):   # we create a method which will return a list of positions(coords) for each point(21)
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]   # getting lm's for one particular hand
            for id, lm in enumerate(myHand.landmark):
                #print(id,lm)
                height, width, channels = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)   # getting coordinates as pixels(dec -> pixels)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    if id % 4 == 0:
                        cv2.circle(img, (cx, cy), 10, (255,0,255), cv2.FILLED)
        return lmList



def main():
    prev_time = 0
    current_time = 0
    cap = cv2.VideoCapture(0)

    detector = handDetector()    # default parameters already given to the class above

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            for i in range(len(lmList)):
                print(lmList[i])

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()