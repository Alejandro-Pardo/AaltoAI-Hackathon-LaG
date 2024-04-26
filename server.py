from people_move_heatmap import PeopleMovementHeatmap
from camera_class import Camera
from file_camera import FileCamera
import time
import cv2
import numpy as np
import MakFormer


class Server:

    def __init__(self):
        self.camera = FileCamera("IMG_1081.MP4", (1280, 720))
        print("Getting Shape")
        shape = self.camera.get_shape()
        print(shape)
        self.people_movement_heatmap = PeopleMovementHeatmap(shape)
        self.maskformer = MakFormer.MaskFormer(shape)
        self.fps = 1
        self.writer = video_writer = cv2.VideoWriter(
            "heatmap_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 3, (shape[1], shape[0])
        )

    def loop(self):
        last_frame = time.time()

        for i in range(30*3):
            print("loop")
            image = self.camera.get_image()
            heatmap = self.people_movement_heatmap.gen_heat(image)
            mask = self.maskformer.gen_heat(image)
            heatmap_image = np.copy(image).astype(np.float64)
            heatmap_image[:,:] *= 0.1
            heatmap_image[:,:,2] = np.zeros_like(heatmap)
            heatmap_image[:,:,2] += mask // 2
            heatmap_image[:,:,2] += heatmap // 2
            heatmap_image[:,:,2] = np.clip(heatmap_image[:,:,2], 0, 255).astype(np.uint8)

            self.display(heatmap_image.astype(np.uint8))
            current_time = time.time()
            time_past = current_time - last_frame
            time_to_sleep = 1 / self.fps - time_past
            #if time_to_sleep > 0:
            #    time.sleep(time_to_sleep)
            last_frame = time.time()
        self.writer.release()

    def display(self, image):
        print("Displaying Heatmap")
        self.writer.write(image)


server = Server()
server.loop()
