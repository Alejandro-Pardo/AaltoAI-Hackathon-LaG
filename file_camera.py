import cv2

class FileCamera():

    def __init__(self, filename, predefined_size = None) -> None:
        self.filename = filename
        self.cap = cv2.VideoCapture(self.filename)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if predefined_size is not None:
            self.predefined_size = predefined_size
        self.current_second = 0


    def get_shape(self):
        if hasattr(self, 'predefined_size'):
            return self.predefined_size
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)

    def get_image(self):
        frame_number = self.current_second * self.fps
        self.current_second += 0.2
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            if hasattr(self, 'predefined_size'):
                frame = cv2.resize(frame, self.predefined_size[:2])
            return frame
        return None
