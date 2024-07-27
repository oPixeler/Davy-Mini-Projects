import cv2
import numpy as np

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

start_coord = None

cap = cv2.VideoCapture(0)

# Camera calibration parameters
focal_length = 500  # The focal length of your camera lens in pixels
known_width = 3  # The known width of the object you're tracking in centimeters

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_hsv = (40, 50, 50)
    upper_hsv = (80, 255, 255)

    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        pen_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(pen_contour)

        pen_center = (int(x + w / 2), int(y + h / 2))

        pen_tip = (pen_center[0], y)

        # Calculate the estimated distance
        distance = (known_width * focal_length) / w  # Distance in centimeters

        cv2.circle(frame, pen_tip, 5, (0, 0, 255), -1)

        if start_coord is not None:
            cv2.line(canvas, start_coord, pen_tip, (0, 255, 0), 2)

        start_coord = pen_tip

        cv2.putText(frame, f"Distance: {distance:.1f} cm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Video', frame)
    cv2.imshow('Canvas', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("PROJEECTS.png", canvas)
        break

cap.release()

cv2.destroyAllWindows()

'''import cv2
import numpy as np
import imutils
import urllib.request

# Create blank canvases for each input
canvas_laptop = np.zeros((480, 640, 3), dtype=np.uint8)
canvas_phone = np.zeros((480, 640, 3), dtype=np.uint8)

# Set the start coordinates for each input
start_coord_laptop = None
start_coord_phone = None

# Set the IP webcam address for the phone camera
phone_cam_address = 'http://192.168.1.62:8080/video'

# Set the desired camera resolution (optional)
frame_width = 640
frame_height = 480

# Define the lower and upper thresholds for green detection in HSV color space
lower_hsv = (40, 50, 50)
upper_hsv = (80, 255, 255)

# Set the focal length of the camera (replace with the actual value)
focal_length = 100

# Initialize the video stream for the laptop camera
cap_laptop = cv2.VideoCapture(0)
cap_laptop.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap_laptop.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Initialize the video stream for the phone camera
stream = urllib.request.urlopen(phone_cam_address)

# Create separate windows for each input
cv2.namedWindow("Laptop Camera")
cv2.namedWindow("Phone Camera")

while True:
    # Read frames from the laptop camera
    ret_laptop, frame_laptop = cap_laptop.read()

    if not ret_laptop:
        break

    frame_laptop = imutils.resize(frame_laptop, width=frame_width)

    # Convert the frame to grayscale for smoothing
    gray_frame_laptop = cv2.cvtColor(frame_laptop, cv2.COLOR_BGR2GRAY)

    blurred_frame_laptop = cv2.GaussianBlur(gray_frame_laptop, (5, 5), 0)

    frame_phone_bytes = stream.read()
    frame_phone = np.frombuffer(frame_phone_bytes, dtype=np.uint8)

    frame_phone = np.reshape(frame_phone, (frame_height, frame_width, 3))
    frame_phone = imutils.resize(frame_phone, width=frame_width)

    hsv_frame_laptop = cv2.cvtColor(frame_laptop, cv2.COLOR_BGR2HSV)
    hsv_frame_phone = cv2.cvtColor(frame_phone, cv2.COLOR_BGR2HSV)

    mask_laptop = cv2.inRange(hsv_frame_laptop, lower_hsv, upper_hsv)
    mask_phone = cv2.inRange(hsv_frame_phone, lower_hsv, upper_hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_laptop = cv2.morphologyEx(mask_laptop, cv2.MORPH_OPEN, kernel)
    mask_laptop = cv2.morphologyEx(mask_laptop, cv2.MORPH_CLOSE, kernel)
    mask_phone = cv2.morphologyEx(mask_phone, cv2.MORPH_OPEN, kernel)
    mask_phone = cv2.morphologyEx(mask_phone, cv2.MORPH_CLOSE, kernel)

    contours_laptop, _ = cv2.findContours(mask_laptop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_phone, _ = cv2.findContours(mask_phone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours_laptop) > 0:
        pen_contour_laptop = max(contours_laptop, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(pen_contour_laptop)

        pen_center_laptop = (int(x + w / 2), int(y + h / 2))

        pen_tip_laptop = (pen_center_laptop[0], y)

        distance_laptop = focal_length * 4 / w 

        cv2.circle(canvas_laptop, pen_tip_laptop, 5, (0, 255, 0), -1)

        cv2.rectangle(frame_laptop, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if start_coord_laptop is not None:
            cv2.line(frame_laptop, start_coord_laptop, pen_center_laptop, (0, 0, 255), 2)

        start_coord_laptop = pen_center_laptop

    else:
        start_coord_laptop = None

    if len(contours_phone) > 0:
        pen_contour_phone = max(contours_phone, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(pen_contour_phone)

        pen_center_phone = (int(x + w / 2), int(y + h / 2))

        pen_tip_phone = (pen_center_phone[0], y)

        distance_phone = focal_length * 4 / w  

        cv2.circle(canvas_phone, pen_tip_phone, 5, (0, 255, 0), -1)

        cv2.rectangle(frame_phone, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if start_coord_phone is not None:
            cv2.line(frame_phone, start_coord_phone, pen_center_phone, (0, 0, 255), 2)

        start_coord_phone = pen_center_phone

    else:
        start_coord_phone = None

    cv2.imshow("Laptop Camera", frame_laptop)
    cv2.imshow("Phone Camera", frame_phone)

    cv2.imshow("Laptop Canvas", canvas_laptop)
    cv2.imshow("Phone Canvas", canvas_phone)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_laptop.release()
cv2.destroyAllWindows()'''