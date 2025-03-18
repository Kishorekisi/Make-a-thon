import cv2
import easyocr
import time

def realtime_ocr():
    """Performs real-time text detection and extraction using webcam."""
    try:
        # Initialize the OCR reader
        reader = easyocr.Reader(['en'])  # You can specify languages here

        # Open the default webcam (usually camera index 0)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            # Convert the frame to grayscale (optional, but can improve OCR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Perform OCR on the frame
            start_time = time.time()
            results = reader.readtext(gray)  # Or use the color frame: reader.readtext(frame)
            end_time = time.time()
            ocr_time = end_time - start_time
            print(f"OCR processing time: {ocr_time:.2f} seconds")

            # Draw bounding boxes and extracted text on the frame
            for (bbox, text, prob) in results:
                # Get the coordinates of the bounding box
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                br = (int(br[0]), int(br[1]))

                # Draw the bounding box
                cv2.rectangle(frame, tl, br, (0, 255, 0), 2)

                # Put the detected text above the bounding box
                cv2.putText(frame, text, (tl[0], tl[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f"Detected: '{text}' with confidence {prob:.2f}")

            # Display the frame with detections
            cv2.imshow('Real-time OCR', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and destroy all windows
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the real-time OCR
if __name__ == "__main__":
    realtime_ocr()