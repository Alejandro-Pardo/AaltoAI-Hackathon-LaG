from transformers import (
    DetrConfig,
    DetrModel,
    AutoImageProcessor,
    DetrForObjectDetection,
)
from PIL import Image
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


class PeopleMovementHeatmap:

    def __init__(self, image_shape, scale=1) -> None:
        self.image_shape = image_shape
        self.heatmap = np.zeros((image_shape[0], image_shape[1]))
        self.people = {}
        self.old_people = {}

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.heat_per_person = 1
        self.heat_radius = int(100 * scale)
        self.heat_per_distance = 1
        self.heat_per_knife = 200
        self.heat_decay = 0.9
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50"
        ).to(self.device)
        self.image_processor = AutoImageProcessor.from_pretrained(
            "facebook/detr-resnet-50"
        )

    def track_people(self, image):
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs.to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([(image.shape[0], image.shape[1])])
        results = self.image_processor.post_process_object_detection(
            outputs, threshold=0.9, target_sizes=target_sizes
        )[0]

        humans = []
        knifes = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            if label.item() == 9:
                knifes.append(np.array([int(round(i)) for i in box.tolist()])[:2])
            if label.item() != 1:
                continue
            box = np.array([int(round(i)) for i in box.tolist()])

            humans.append(box)

        humans = np.array(humans)
        human_middle = humans[:, :2] + (humans[:, 2:] - humans[:, :2]) / 2

        new_people = {}
        for human in human_middle:
            min_dist = float("inf")
            closest_person = None
            closest_id = None
            for id, person in self.people.items():
                dist = (
                    (person[0] - human[0]) ** 2 + (person[1] - human[1]) ** 2
                ) ** 0.5
                if dist < min_dist and id not in new_people.keys() and dist < 100:
                    min_dist = dist
                    closest_person = person
                    closest_id = id
            if closest_person is not None:
                new_people[closest_id] = human
            else:
                new_people[
                    max(
                        max(self.people.keys(), default=0),
                        max(new_people.keys(), default=0),
                    )
                    + 1
                ] = human

        return new_people, knifes

    def gen_heat(self, image):
        self.people, self.knifes = self.track_people(image)

        self.heatmap *= self.heat_decay

        self._gen_people_heat()
        self._gen_movement_heat()

        for knife in self.knifes:
            x, y = knife
            x, y = int(x), int(y)
            self._gen_heat_radius(self.heat_per_knife, x, y)

        self.old_people = self.people
        return np.clip(self.heatmap, 0, 255).astype(np.uint8)

    def _gen_people_heat(self):
        for person in self.people.values():
            x, y = person
            x, y = int(x), int(y)

            self._gen_heat_radius(self.heat_per_person, x, y)

    def _gen_movement_heat(self):
        for id, person in self.people.items():
            x, y = person
            x, y = int(x), int(y)
            if id in self.old_people:
                old_x, old_y = self.old_people[id]
                old_x, old_y = int(old_x), int(old_y)
                distance_to_old = ((x - old_x) ** 2 + (y - old_y) ** 2) ** 0.5

                self._gen_heat_radius(self.heat_per_distance * distance_to_old, x, y)

    def _gen_heat_radius(self, heat, x, y):
        start = max(0, x - self.heat_radius), max(0, y - self.heat_radius)
        end = min(self.image_shape[1], x + self.heat_radius), min(
            self.image_shape[0], y + self.heat_radius
        )

        x = np.arange(start[0], end[0])
        y = np.arange(start[1], end[1])
        xx, yy = np.meshgrid(x, y)

        matrix = np.dstack([xx, yy])
        distance = np.linalg.norm(matrix - (x, y), axis=2)
        self.heatmap[start[1] : end[1], start[0] : end[0]] += heat * (
            1 - distance / self.heat_radius
        )
