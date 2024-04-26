import cv2
import threading

class Camera:
    def __init__(self, rtsp_url="rtsp://lmango:lisan-ai-gaib@192.168.33.61:554/stream1"):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url)
        self.frame_lock = threading.Lock()
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return
        self._start_camera_thread()
        
    def _start_camera_thread(self):
        camera_thread = threading.Thread(target=self._fetch_feed)
        camera_thread.start()

    def _fetch_feed(self):
        started = False
        try:
            while True:
                _, frame = self.cap.read()
                if not started:
                    print("Camera started")
                    started = True
                if frame is None:
                    break
                self.frame_lock.acquire()
                self.frame = frame;
                self.frame_lock.release()
        except cv2.error as e:
            print("Error:", e)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Stream closed")

    def get_shape(self):
        shape = None
        self.frame_lock.acquire()
        if self.frame is not None:
            shape = self.frame.shape
        self.frame_lock.release()
        return shape

    def get_image(self):
        frame = None
        self.frame_lock.acquire()
        if self.frame is not None:
            frame = self.frame
        self.frame_lock.release()
        return frame

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