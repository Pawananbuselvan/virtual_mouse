import math
import time
import autopy
import cv2
import HandTracker as ht
import numpy as np

# Configuration parameters
width, height = 640, 480
frameReduction = 100  # Frame boundary margin
smooth = 7  # Smoothing factor
click_threshold = 40  # Max distance (px) between fingers to trigger a click

plocX, plocY = 0, 0
clocX, clocY = 0, 0

detector = ht.HandDetector(maxHands=1)

video = cv2.VideoCapture(0)
video.set(3, width)
video.set(4, height)

ptime = 0
wscr, hscr = autopy.screen.size()  # Fixed: width first, height second

while True:
    ret, frame = video.read()
    if not ret:
        break

    # 1. Flip camera frame for intuitive right/left mirroring
    frame = cv2.flip(frame, 1)

    # 2. Detect hands and landmarks
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame)

    # Draw active screen boundaries on frame
    cv2.rectangle(
        frame,
        (frameReduction, frameReduction),
        (width - frameReduction, height - frameReduction),
        (255, 0, 255),
        2,
    )

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index tip
        x2, y2 = lmList[12][1:]  # Middle tip

        move = detector.fingerUp(8) and not detector.fingerUp(12)
        click = detector.fingerUp(8) and detector.fingerUp(12)

        # MODE 1: MOVING CURSOR
        if move:
            # Convert coordinates with boundary limits
            x3 = np.interp(
                x1, (frameReduction, width - frameReduction), (0, wscr)
            )
            y3 = np.interp(
                y1, (frameReduction, height - frameReduction), (0, hscr)
            )

            # Smooth cursor movement
            clocX = plocX + (x3 - plocX) / smooth
            clocY = plocY + (y3 - plocY) / smooth

            try:
                autopy.mouse.move(clocX, clocY)
            except ValueError:
                pass  # Safely ignore minor out-of-bounds rounding glitches

            plocX, plocY = clocX, clocY
            cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

        # MODE 2: CLICKING
        if click:
            length = np.hypot(x2 - x1, y2 - y1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if length < click_threshold:
                cv2.circle(frame, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
                time.sleep(0.15)  # Prevents rapid infinite clicking

    # Calculate and display FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime) if (ctime - ptime) > 0 else 0
    ptime = ctime

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (255, 0, 0),
        2,
    )

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()