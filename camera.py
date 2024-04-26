import cv2

def display_rtsp_stream(rtsp_url):
    print("Connecting to RTSP stream:", rtsp_url)
    # Create a video capture object with the RTSP URL
    cap = cv2.VideoCapture(rtsp_url)
    print("VideoCapture object created:", cap.isOpened())

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    try:
        # Loop to continuously fetch frames from the RTSP stream
        while True:
            print("Reading frame...")
            ret, frame = cap.read()
            print("Frame read:", ret)
            # If frame is read correctly ret is True
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting...")
                break

            # Display the resulting frame
            cv2.imshow('RTSP Stream', frame)
            print("Frame displayed")
            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except cv2.error as e:
        print("Error:", e)
    finally:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        print("Stream closed")

rtsp_url = "rtsp://lmango:lisan-ai-gaib@192.168.33.61:554/stream1"
display_rtsp_stream(rtsp_url)