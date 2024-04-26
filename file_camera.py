import cv2

class FileCamera():

    def __init__(self, filename) -> None:
        self.filename = filename
        self.cap = cv2.VideoCapture(self.filename)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_second = 0


    def get_shape(self):
        return (self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), 3)

    def get_image(self):
        frame_number = self.current_second * self.fps
        self.current_second += 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
