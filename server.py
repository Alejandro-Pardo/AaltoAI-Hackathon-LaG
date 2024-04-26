from people_move_heatmap import PeopleMovementHeatmap
from camera_class import Camera
from file_camera import FileCamera
import time
import cv2
import numpy as np
import MakFormer


class Server:

    def __init__(self):
        self.camera = FileCamera("IMG_1081.MP4")
        time.sleep(10)
        print("Getting Shape")
        shape = self.camera.get_shape()
        print(shape)
        self.people_movement_heatmap = PeopleMovementHeatmap(shape)
        self.maskformer = MakFormer.MaskFormer(shape)
        self.fps = 1
        self.writer = video_writer = cv2.VideoWriter(
            "heatmap_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 1, (shape[1], shape[0])
        )

    def loop(self):
        last_frame = time.time()

        for i in range(20):
            print("loop")
            image = self.camera.get_image()
            heatmap = self.people_movement_heatmap.gen_heat(image)
            mask = self.maskformer.gen_heat(image)
            heatmap_image = np.zeros_like(image)
            heatmap_image[:,:,0] = mask
            self.display(heatmap_image)
            current_time = time.time()
            time_past = current_time - last_frame
            time_to_sleep = 1 / self.fps - time_past
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            last_frame = time.time()
        self.writer.release()

    def display(self, image):
        print("Displaying Heatmap")
        self.writer.write(image)


server = Server()
server.loop()
