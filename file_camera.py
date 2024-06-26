import cv2

class FileCamera():

    def __init__(self, filename, predefined_size = None) -> None:
        self.filename = filename
        self.cap = cv2.VideoCapture(self.filename)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if predefined_size is not None:
            self.predefined_size = predefined_size
        self.current_second = 0
        self.frame_number = 0


    def get_shape(self):
        if hasattr(self, 'predefined_size'):
            return (self.predefined_size[1], self.predefined_size[0], 3)
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)

    def get_image(self):
        frame_number = int(self.current_second * self.fps)
        self.current_second += 1/3
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            if hasattr(self, 'predefined_size'):
                frame = cv2.resize(frame, self.predefined_size)
            return frame
        return None
    
    def get_image_one_by_one(self):
        frame_number = self.frame_number
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        self.frame_number += 1
        if ret:
            if hasattr(self, 'predefined_size'):
                frame = cv2.resize(frame, self.predefined_size)
            return frame
        return None
