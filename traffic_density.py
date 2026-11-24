import cv2
from ultralytics import YOLO

# -----------------------------------
# 1. Load YOLO model
# -----------------------------------
model = YOLO("yolo11s.pt")

# -----------------------------------
# 2. Open camera
# -----------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# -----------------------------------
# 3. COCO class IDs
# -----------------------------------
PERSON = 0
CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7

# -----------------------------------
# 4. Start detection
# -----------------------------------
while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    # -----------------------------------
    # 5. YOLO detection
    # -----------------------------------
    results = model(
        frame,
        conf=0.20,
        imgsz=640,
        verbose=False
    )

    # -----------------------------------
    # 6. Initialize counters
    # -----------------------------------
    person_count = 0
    car_count = 0
    motorcycle_count = 0
    bus_count = 0
    truck_count = 0

    # -----------------------------------
    # 7. Count detected objects
    # -----------------------------------
    for box in results[0].boxes:

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

    # -----------------------------------
    # 8. Calculate total vehicles
    # -----------------------------------
    total_vehicles = (
        car_count
        + motorcycle_count
        + bus_count
        + truck_count
    )

    # -----------------------------------
    # 9. Traffic density
    # -----------------------------------
    if total_vehicles <= 5:
        density = "LOW"

    elif total_vehicles <= 15:
        density = "MEDIUM"

    else:
        density = "HIGH"

    # -----------------------------------
    # 10. Draw bounding boxes
    # -----------------------------------
    annotated_frame = results[0].plot()

    # -----------------------------------
    # 11. Display information
    # -----------------------------------

    cv2.putText(
        annotated_frame,
        f"Humans: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Cars: {car_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Motorcycles: {motorcycle_count}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Buses: {bus_count}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Trucks: {truck_count}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Total Vehicles: {total_vehicles}",
        (20, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Traffic Density: {density}",
        (20, 265),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    # -----------------------------------
    # 12. Display window
    # -----------------------------------
    cv2.imshow(
        "Traffic Density Monitoring",
        annotated_frame
    )

    # -----------------------------------
    # 13. Press Q to quit
    # -----------------------------------
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------------
# 14. Release camera
# -----------------------------------
cap.release()
cv2.destroyAllWindows()