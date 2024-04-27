import numpy as np
import skimage.measure


class PropGen():
    def __init__(self, shape) -> None:
        self.shape = (shape[1], shape[0])

    def gen_properties(self, heat, num_people):
        mean_heat = np.mean(heat)
        std_heat = np.std(heat)
        max_heat = np.max(heat)
        min_heat = np.min(heat)
        num_people = num_people
        total_heat = np.sum(heat)
        heat_per_person = total_heat / num_people
        reduced_heatmap = skimage.measure.block_reduce(heat, (20, 20), np.mean)
        x0, y0 = reduced_heatmap.shape
        x1, y1 = 0,0
        for x in range(reduced_heatmap.shape[0]):
            for y in range(reduced_heatmap.shape[1]):
                if reduced_heatmap[x, y] - mean_heat > 0.5 * std_heat:
                    x0 = min(x0, x)
                    y0 = min(y0, y)
                    x1 = max(x1, x)
                    y1 = max(y1, y)
        
        return mean_heat, max_heat, total_heat, heat_per_person, x0, y0, x1, y1