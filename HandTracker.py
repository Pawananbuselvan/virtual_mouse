import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, frame, draw=True):   
        frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRgb) 
        #print(self.results.multi_hand_landmarks)   

        if draw:
            if self.results.multi_hand_landmarks:
                for self.handLms in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(frame, self.handLms,self.mpHands.HAND_CONNECTIONS) 
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        self.lmlist=[]  
        if self.results.multi_hand_landmarks: 
            for self.handLms in self.results.multi_hand_landmarks:   
                for id,lm in enumerate(self.handLms.landmark):
                    h,w,c =frame.shape
                    cx,cy=int(lm.x*w),int(lm.y*h)
                    self.lmlist.append([id,cx,cy])
        return self.lmlist
    def highlightFinger(self, frame,i=4, handNo=0):
        if self.results.multi_hand_landmarks: 
            for self.handLms in self.results.multi_hand_landmarks:   
                for id,lm in enumerate(self.handLms.landmark):
                    h,w,c =frame.shape
                    cx,cy=int(lm.x*w),int(lm.y*h)
                    if id==i:
                        cv2.circle(frame,(cx,cy),15,(255,0,0),cv2.FILLED)
                        cv2.rectangle(frame,(cx-7,cy-7),(cx+7,cy+7),(255,0,255),cv2.FILLED)
        return frame
    
    def fingerUp(self,i):
        if self.lmlist[i][2]<self.lmlist[i-1][2]:
            return True
        


def main():
    video = cv2.VideoCapture(0)
    ptime=0
    detector = HandDetector()

    while True:
        ret, frame = video.read()
        frame = detector.findHands(frame)
        lmList= detector.findPosition(frame)
        frame = detector.highlightFinger(frame,i=0)

        ctime = time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        cv2.imshow("cam", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()