from people_move_heatmap import PeopleMovementHeatmap
from camera_class import Camera
import time
import cv2
import numpy as np



class Server():

    def __init__(self):
        self.camera = Camera()
        self.people_movement_heatmap = PeopleMovementHeatmap(self.camera.get_shape())
        self.fps = 1

    def loop(self):
        last_frame = time.time()
        while True:
            print("loop")
            image = self.camera.get_image()
            heatmap = self.people_movement_heatmap.gen_heat(image)
            heatmap_image = image.copy()
            heatmap_image[:,:,0] += heatmap
            heatmap_image = np.clip(heatmap_image, 0, 255)
            self.display(heatmap_image)
            current_time = time.time()
            time_past = current_time - last_frame
            time_to_sleep = 1 / self.fps - time_past
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            last_frame = time.time()

    def display(self, image):
        cv2.imshow('Heatmap', image)

server = Server()
server.loop()

            
