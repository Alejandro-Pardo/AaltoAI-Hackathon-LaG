import cv2

class Camera:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return

    def get_shape(self):
        ret, frame = self.cap.read()
        if ret:
            return frame.shape
        else:
            print("Error: Can't receive frame (stream end?). Exiting...")
            return None

    def get_image(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            print("Error: Can't receive frame (stream end?). Exiting...")
            return None

    def display_rtsp_stream(self):
        try:
            while True:
                frame = self.get_image()
                if frame is None:
                    break
                cv2.imshow('RTSP Stream', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except cv2.error as e:
            print("Error:", e)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Stream closed")

rtsp_url = "rtsp://lmango:lisan-ai-gaib@192.168.33.61:554/stream1"
camera = Camera(rtsp_url)
camera.display_rtsp_stream()