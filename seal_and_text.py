import cv2
import easyocr
from ultralytics import YOLO

def extract_row_text(table_width, cell_width, seal_x, frame, reader):
    """
    Extracts text from the row containing the detected seal.

    Args:
        table_width (int): Total width of the table in pixels.
        cell_width (int): Width of each cell in the table in pixels.
        seal_x (int): X-coordinate of the detected seal (center of the bounding box).
        frame (numpy.ndarray): The input frame from the camera.
        reader (easyocr.Reader): The EasyOCR reader object.

    Returns:
        str: The extracted text from the row, or None if an error occurs.
    """

    # Determine the row number
    row_number = seal_x // cell_width

    # Calculate the start and end x-coordinates of the row
    row_start_x = row_number * cell_width
    row_end_x = (row_number + 1) * cell_width

    # Crop the row from the frame
    row_image = frame[:, row_start_x:row_end_x]

    # Perform OCR on the row image
    results = reader.readtext(row_image)

    # Extract the text from the OCR results
    row_text = ""
    for (bbox, text, prob) in results:
        row_text += text + " "  # Concatenate the text from all detected words

    return row_text.strip()  # Remove leading/trailing whitespace


if __name__ == "__main__":
    # Load YOLO model
    model = YOLO('trained_model.pt')

    # Load EasyOCR reader
    reader = easyocr.Reader(['en'])  # Adjust language and GPU usage as needed

    # Table parameters (adjust these based on your table)
    table_width = 640  # Example: Total width of the table in pixels
    cell_width = 80   # Example: Width of each cell in pixels

    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform object detection
        results = model.predict(source=frame, conf=0.5, imgsz=640, show=False)  # Don't show the YOLO output directly

        # Process the detection results
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert to integers
                seal_x = (x1 + x2) // 2  # Calculate the center x-coordinate of the seal

                # Extract row text
                row_text = extract_row_text(table_width, cell_width, seal_x, frame, reader)

                if row_text:
                    print(f"Seal detected. Row text: {row_text}")

                    # Draw a rectangle around the detected seal (optional)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Seal", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw the row bounding box (optional)
                row_start_x = (seal_x // cell_width) * cell_width
                row_end_x = (seal_x // cell_width + 1) * cell_width
                cv2.rectangle(frame, (row_start_x, 0), (row_end_x, frame.shape[0]), (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Real-time OCR', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()