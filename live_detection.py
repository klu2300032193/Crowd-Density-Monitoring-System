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
# 3. Required classes
# -----------------------------------
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

# -----------------------------------
# 4. Start detection
# -----------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    # -----------------------------------
    # 5. Detect ONLY required classes
    # -----------------------------------
    results = model(
        frame,
        conf=0.30,
        imgsz=640,
        classes=REQUIRED_CLASSES,
        verbose=False
    )

    result = results[0]

    # -----------------------------------
    # 6. Counters
    # -----------------------------------
    person_count = 0
    car_count = 0
    motorcycle_count = 0
    bus_count = 0
    truck_count = 0

    # -----------------------------------
    # 7. Copy original frame
    # -----------------------------------
    annotated_frame = frame.copy()

    # -----------------------------------
    # 8. Process detections
    # -----------------------------------
    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # --------------------------------
        # Person
        # --------------------------------
        if class_id == PERSON:

            person_count += 1
            label = f"Person {confidence:.2f}"

        # --------------------------------
        # Car
        # --------------------------------
        elif class_id == CAR:

            car_count += 1
            label = f"Car {confidence:.2f}"

        # --------------------------------
        # Motorcycle
        # --------------------------------
        elif class_id == MOTORCYCLE:

            motorcycle_count += 1
            label = f"Motorcycle {confidence:.2f}"

        # --------------------------------
        # Bus
        # --------------------------------
        elif class_id == BUS:

            bus_count += 1
            label = f"Bus {confidence:.2f}"

        # --------------------------------
        # Truck
        # --------------------------------
        elif class_id == TRUCK:

            truck_count += 1
            label = f"Truck {confidence:.2f}"

        else:
            continue

        # --------------------------------
        # Draw bounding box
        # --------------------------------
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # --------------------------------
        # Draw label
        # --------------------------------
        cv2.putText(
            annotated_frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # -----------------------------------
    # 9. Total vehicles
    # -----------------------------------
    total_vehicles = (
        car_count
        + motorcycle_count
        + bus_count
        + truck_count
    )

    # -----------------------------------
    # 10. Crowd density
    # -----------------------------------
    if person_count <= 4:
        crowd_density = "LOW"

    elif person_count <= 9:
        crowd_density = "MEDIUM"

    else:
        crowd_density = "HIGH"

    # -----------------------------------
    # 11. Traffic density
    # -----------------------------------
    if total_vehicles <= 5:
        traffic_density = "LOW"

    elif total_vehicles <= 15:
        traffic_density = "MEDIUM"

    else:
        traffic_density = "HIGH"

    # -----------------------------------
    # 12. Display information
    # -----------------------------------

    cv2.putText(
        annotated_frame,
        f"People: {person_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Cars: {car_count}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Motorcycles: {motorcycle_count}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Buses: {bus_count}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Trucks: {truck_count}",
        (20, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Total Vehicles: {total_vehicles}",
        (20, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Crowd Density: {crowd_density}",
        (20, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Traffic Density: {traffic_density}",
        (20, 265),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Crowd and Traffic Monitoring",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()