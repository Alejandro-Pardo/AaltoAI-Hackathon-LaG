from people_move_heatmap import PeopleMovementHeatmap
from prop_gen import PropGen
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
        self.prop_gen = PropGen(shape)
        self.fps = 1
        self.writer = video_writer = cv2.VideoWriter(
            "heatmap_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 3, (shape[1], shape[0])
        )

    def loop(self):
        last_frame = time.time()

        for i in range(15*3):
            print("loop")
            image = self.camera.get_image()
            heatmap = self.people_movement_heatmap.gen_heat(image)
            num_people = len(self.people_movement_heatmap.people)
            mask = self.maskformer.gen_heat(image)
            heatmap_image = np.copy(image).astype(np.float64)
            heatmap = heatmap // 2
            heatmap += mask // 2
            heatmap_image[:,:] *= 0.1
            heatmap_image[:,:,2] = np.clip(heatmap, 0, 255).astype(np.uint8)
            props = self.prop_gen.gen_properties(heatmap, num_people)
            mean_heat, max_heat, total_heat, heat_per_person, x0, y0, x1, y1 = props
            if x0 < x1 and y0 < y1:
                cv2.rectangle(heatmap_image, (y0, x0), (y1, x1), (0, 255, 0), 2)
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
