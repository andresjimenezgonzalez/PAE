import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

def max_min(xMax, xMin, yMax, yMin, xFinger, yFinger):
	if xFinger > xMax:
		xMax = xFinger
	elif xFinger < xMin:
		xMin = xFinger

	if yFinger > yMax:
		yMax = yFinger
	elif yFinger < yMin:
		yMin = yFinger

	return xMax, xMin, yMax, yMin

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.5) as hands:

	while True: 
		ret, frame = cap.read()
		if ret == False:
			break

		height, width, _ = frame.shape
		frame = cv2.flip(frame,1)
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		results = hands.process(frame_rgb)

		#cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        #cv2.putText(frame, hand_handedness.classification[0].label, (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
		if results.multi_hand_landmarks is not None and results.multi_handedness is not None:
			for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
				xMiddleFingerMcp = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * width)
				yMiddleFingerMcp = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * height)

				xMax = xMin = xMiddleFingerMcp
				yMax = yMin = yMiddleFingerMcp

				xThumbTip = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
				yThumbTip = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)

				xMax, xMin, yMax, yMin = max_min(xMax, xMin, yMax, yMin, xThumbTip, yThumbTip)

				xIndexFingerTip = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
				yIndexFingerTip = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)

				xMax, xMin, yMax, yMin = max_min(xMax, xMin, yMax, yMin, xIndexFingerTip, yIndexFingerTip)

				xMiddleFingerTip = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width)
				yMiddleFingerTip = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height)

				xMax, xMin, yMax, yMin = max_min(xMax, xMin, yMax, yMin, xMiddleFingerTip, yMiddleFingerTip)

				xRingFingerTip = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * width)
				yRingFingerTip = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * height)

				xMax, xMin, yMax, yMin = max_min(xMax, xMin, yMax, yMin, xRingFingerTip, yRingFingerTip)

				xPinkyTip = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * width)
				yPinkyTip = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * height)

				xMax, xMin, yMax, yMin = max_min(xMax, xMin, yMax, yMin, xPinkyTip, yPinkyTip)

				xIndexFingerMcp = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * width)
				yIndexFingerMcp = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * height)

				xMax, xMin, yMax, yMin = max_min(xMax, xMin, yMax, yMin, xIndexFingerMcp, yIndexFingerMcp)

				xMiddleFingerMcp = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * width)
				yMiddleFingerMcp = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * height)

				xMax, xMin, yMax, yMin = max_min(xMax, xMin, yMax, yMin, xMiddleFingerMcp, yMiddleFingerMcp)

				xRingFingerMcp = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * width)
				yRingFingerMcp = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * height)

				xMax, xMin, yMax, yMin = max_min(xMax, xMin, yMax, yMin, xRingFingerMcp, yRingFingerMcp)

				xPinkyMcp = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * width)
				yPinkyMcp = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * height)

				xMax, xMin, yMax, yMin = max_min(xMax, xMin, yMax, yMin, xPinkyMcp, yPinkyMcp)

				xIndexFingerPip = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * width)
				yIndexFingerPip = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * height)

				xMax, xMin, yMax, yMin = max_min(xMax, xMin, yMax, yMin, xIndexFingerPip, yIndexFingerPip)

				xMiddleFingerPip = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * width)
				yMiddleFingerPip = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * height)

				xMax, xMin, yMax, yMin = max_min(xMax, xMin, yMax, yMin, xMiddleFingerPip, yMiddleFingerPip)

				xRingFingerPip = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * width)
				yRingFingerPip = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * height)

				xMax, xMin, yMax, yMin = max_min(xMax, xMin, yMax, yMin, xRingFingerPip, yRingFingerPip)

				xPinkyPip = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * width)
				yPinkyPip = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * height)

				xMax, xMin, yMax, yMin = max_min(xMax, xMin, yMax, yMin, xPinkyPip, yPinkyPip)

				xWrist = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
				yWrist = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)

				xMax, xMin, yMax, yMin = max_min(xMax, xMin, yMax, yMin, xWrist, yWrist)

				yIndexFingerDip = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * height)

				if yIndexFingerTip - yThumbTip < 50 and yThumbTip < yIndexFingerTip < yMiddleFingerTip < yRingFingerTip < yPinkyTip and ((xIndexFingerTip < xIndexFingerPip and xMiddleFingerTip < xMiddleFingerPip and xRingFingerTip < xRingFingerPip and xPinkyTip < xPinkyPip and hand_handedness.classification[0].label == "Left") or	(xIndexFingerTip > xIndexFingerPip and xMiddleFingerTip > xMiddleFingerPip and xRingFingerTip > xRingFingerPip and xPinkyTip > xPinkyPip and hand_handedness.classification[0].label == "Right")):
					text = "A"
					cv2.putText(frame, text, (xMin - 5, yMin - 20), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
					cv2.rectangle(frame, (xMin - 10, yMin - 10), (xMax + 10, yMax + 10), (0, 0, 255), 3)
				elif yWrist > yIndexFingerTip and abs(yIndexFingerPip - yIndexFingerDip) < 30 and abs(yThumbTip - yWrist) < 60 and yIndexFingerTip > yIndexFingerPip and yMiddleFingerTip > yMiddleFingerPip and yRingFingerTip > yRingFingerPip and yPinkyTip > yPinkyPip and ((xPinkyPip < xWrist and hand_handedness.classification[0].label == "Left") or (xPinkyPip > xWrist and hand_handedness.classification[0].label == "Right")):
					text = "E"
					cv2.putText(frame, text, (xMin - 5, yMin - 20), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)
					cv2.rectangle(frame, (xMin - 10, yMin - 10), (xMax + 10, yMax + 10), (0, 255, 0), 3)
				elif yPinkyPip - yPinkyTip > 10 and ((xThumbTip < xIndexFingerPip and hand_handedness.classification[0].label == "Left") or (xThumbTip > xIndexFingerPip and hand_handedness.classification[0].label == "Right"))  and yRingFingerTip > yRingFingerPip and yMiddleFingerTip > yMiddleFingerPip and yIndexFingerTip > yIndexFingerPip:
					text = "I"
					cv2.putText(frame, text, (xMin - 5, yMin - 20), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
					cv2.rectangle(frame, (xMin - 10, yMin - 10), (xMax + 10, yMax + 10), (255, 0, 0), 3)
				elif yPinkyPip - yPinkyTip > 10 and yRingFingerPip - yRingFingerTip > 10 and yMiddleFingerPip - yMiddleFingerTip > 10 and abs(xRingFingerTip - xPinkyTip) < 25 and abs(xRingFingerTip - xMiddleFingerTip) < 25 and abs(xThumbTip - xIndexFingerTip) < 20 and abs(yThumbTip - yIndexFingerTip) < 20:
					text = "O"
					cv2.putText(frame, text, (xMin - 5, yMin - 20), 1, 1.3, (0, 128, 255), 1, cv2.LINE_AA)
					cv2.rectangle(frame, (xMin - 10, yMin - 10), (xMax + 10, yMax + 10), (0, 128, 255), 3)
				elif ((xThumbTip < xIndexFingerPip and xMiddleFingerTip < xMiddleFingerPip and xIndexFingerTip > xIndexFingerPip and hand_handedness.classification[0].label == "Left") or (xThumbTip > xIndexFingerPip and xMiddleFingerTip > xMiddleFingerPip and xIndexFingerTip < xIndexFingerPip and hand_handedness.classification[0].label == "Right")) and yPinkyTip > yPinkyPip and yRingFingerTip > yRingFingerPip:
					text = "U"
					cv2.putText(frame, text, (xMin - 5, yMin - 20), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)
					cv2.rectangle(frame, (xMin - 10, yMin - 10), (xMax + 10, yMax + 10), (0, 255, 255), 3)
				else:
					cv2.rectangle(frame, (xMin - 10, yMin - 10), (xMax + 10, yMax + 10), (0, 0, 0), 3)
				mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
				
		cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty("Frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
		cv2.imshow("Frame", frame)

		if cv2.waitKey(1) & 0xFF == 27:
			break

cap.release()
cv2.destroyAllWindows()