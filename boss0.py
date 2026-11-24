import cv2
import time
import serial
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION & SERIAL SETUP
# ==========================================

MODEL_PATH = "yolo11s.pt"

CONFIDENCE = 0.18
IMAGE_SIZE = 960
MAX_DETECTIONS = 200

# Set this to your ESP32 COM port (e.g., 'COM3', 'COM4' on Windows, or '/dev/ttyUSB0' on Linux/Mac)
SERIAL_PORT = ("COM7")
BAUD_RATE = 115200

try:
    esp32 = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)  # Wait for ESP32 boot sequence
    print(f"Connected to ESP32 on {SERIAL_PORT}")
except Exception as e:
    print(f"Warning: Could not connect to ESP32 on {SERIAL_PORT}: {e}")
    esp32 = None

# ==========================================
# 2. LOAD YOLO MODEL
# ==========================================

print("Loading YOLO11s model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

# ==========================================
# 3. OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

# ==========================================
# 4. COCO CLASS IDs
# ==========================================

PERSON = 0
CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7

REQUIRED_CLASSES = [
    PERSON,
    CAR,
    MOTORCYCLE,
    BUS,
    TRUCK
]

# ==========================================
# 5. RUNTIME SETTINGS
# ==========================================

night_mode = False
img_size = IMAGE_SIZE

fps = 0.0
previous_time = time.perf_counter()
last_serial_send_time = 0.0
SERIAL_INTERVAL = 0.5  # Transmit state every 500ms

print("\nControls:")
print("N - Toggle nighttime enhancement")
print("I - Toggle inference size: 640 / 960")
print("Q - Quit")

# ==========================================
# 6. NIGHTTIME IMAGE ENHANCEMENT
# ==========================================

def enhance_night(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

# ==========================================
# 7. MAIN DETECTION LOOP
# ==========================================

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read webcam frame.")
            break

        height, width = frame.shape[:2]

        # 8. PREPROCESSING
        detection_frame = enhance_night(frame) if night_mode else frame

        # 9. YOLO DETECTION ON CPU
        results = model.predict(
            source=detection_frame,
            conf=CONFIDENCE,
            imgsz=img_size,
            classes=REQUIRED_CLASSES,
            max_det=MAX_DETECTIONS,
            device="cpu",
            verbose=False
        )

        result = results[0]

        # 10. COUNT OBJECTS
        person_count = 0
        car_count = 0
        motorcycle_count = 0
        bus_count = 0
        truck_count = 0

        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id == PERSON:
                person_count += 1
            elif class_id == CAR:
                car_count += 1
            elif class_id == MOTORCYCLE:
                motorcycle_count += 1
            elif class_id == BUS:
                bus_count += 1
            elif class_id == TRUCK:
                truck_count += 1

        # 11. TOTAL VEHICLES
        total_vehicles = car_count + motorcycle_count + bus_count + truck_count

        # 12. DENSITY CALCULATIONS & SIGNAL MAPPING
        if person_count <= 4:
            crowd_density = "LOW"
            density_signal = b'L'
        elif person_count <= 9:
            crowd_density = "MEDIUM"
            density_signal = b'M'
        else:
            crowd_density = "HIGH"
            density_signal = b'H'

        if total_vehicles <= 5:
            traffic_density = "LOW"
        elif total_vehicles <= 15:
            traffic_density = "MEDIUM"
        else:
            traffic_density = "HIGH"

        # SEND SIGNAL TO ESP32
        current_time = time.perf_counter()
        if esp32 and (current_time - last_serial_send_time >= SERIAL_INTERVAL):
            try:
                esp32.write(density_signal)
                last_serial_send_time = current_time
            except Exception as e:
                print(f"Serial transmission error: {e}")

        # 13. DRAW BOUNDING BOXES
        annotated_frame = result.plot()

        # 14. CALCULATE FPS
        elapsed = current_time - previous_time
        previous_time = current_time

        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            fps = instant_fps if fps == 0 else (0.85 * fps + 0.15 * instant_fps)

        # 15. INFORMATION PANEL
        lines = [
            (f"People: {person_count}", (255, 255, 0)),
            (f"Cars: {car_count}", (0, 255, 0)),
            (f"Motorcycles: {motorcycle_count}", (0, 255, 0)),
            (f"Buses: {bus_count}", (0, 255, 0)),
            (f"Trucks: {truck_count}", (0, 255, 0)),
            (f"Total Vehicles: {total_vehicles}", (0, 255, 255)),
            (f"Crowd Density: {crowd_density}", (0, 255, 255)),
            (f"Traffic Density: {traffic_density}", (0, 255, 255)),
            (f"FPS: {fps:.1f}", (255, 255, 255)),
            (f"Night Enhancement: {'ON' if night_mode else 'OFF'}", (255, 255, 255)),
            (f"Inference Size: {img_size}", (255, 255, 255)),
            (f"Camera: {width}x{height}", (255, 255, 255))
        ]

        panel_height = 12 + len(lines) * 27
        cv2.rectangle(annotated_frame, (5, 5), (455, panel_height), (25, 25, 25), -1)

        for i, (text, color) in enumerate(lines):
            y = 28 + i * 27
            cv2.putText(annotated_frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)

        # 16. DISPLAY WINDOW
        cv2.imshow("Traffic and Crowd Monitoring", annotated_frame)

        # 17. KEYBOARD CONTROLS
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            night_mode = not night_mode
            print("Night enhancement:", "ON" if night_mode else "OFF")
        elif key == ord("i"):
            img_size = 640 if img_size == 960 else 960
            print("Inference size changed to:", img_size)

finally:
    if esp32:
        esp32.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Serial connection closed. Program stopped.")