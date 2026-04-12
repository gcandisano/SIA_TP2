from __future__ import annotations

import atexit
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from PIL import Image, ImageDraw

from individual import Individual

_executor = None

def get_executor():
    global _executor
    if _executor is None:
        cpus = os.cpu_count() or 1
        _executor = ProcessPoolExecutor(max_workers=cpus)
    return _executor

@atexit.register
def close_executor():
    global _executor
    if _executor is not None:
        _executor.shutdown()

def render(individual: Individual, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))

    for t in individual.triangles:
        # Bounding box calculation
        x_coords = (t.x1, t.x2, t.x3)
        y_coords = (t.y1, t.y2, t.y3)
        
        min_x = max(0, int(min(x_coords)))
        max_x = min(width, int(max(x_coords)) + 1)
        min_y = max(0, int(min(y_coords)))
        max_y = min(height, int(max(y_coords)) + 1)
        
        w, h = max_x - min_x, max_y - min_y
        if w <= 0 or h <= 0:
            continue

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        v = [(t.x1 - min_x, t.y1 - min_y), 
             (t.x2 - min_x, t.y2 - min_y), 
             (t.x3 - min_x, t.y3 - min_y)]
        
        color = (int(t.r), int(t.g), int(t.b), int(t.a * 255))
        draw.polygon(v, fill=color)
        canvas.paste(overlay, (min_x, min_y), overlay)

    return canvas


def compute_fitness(individual: Individual, target: np.ndarray, width: int, height: int) -> float:
    rendered_img = render(individual, width, height).convert("RGB")
    rendered = np.array(rendered_img, dtype=np.float32)
    mse = np.mean(np.square(rendered - target)) / (255.0 ** 2)
    return 1.0 - float(mse)


def _evaluate_single(individual: Individual, target: np.ndarray, width: int, height: int) -> float:
    return compute_fitness(individual, target, width, height)


def evaluate_population(population: list[Individual], target: np.ndarray, width: int, height: int) -> None:
    cpus = os.cpu_count() or 1
    
    if len(population) < 10 or cpus == 1:
        for individual in population:
            individual.fitness = compute_fitness(individual, target, width, height)
    else:
        executor = get_executor()
        eval_func = partial(_evaluate_single, target=target, width=width, height=height)
        results = list(executor.map(eval_func, population))
        
        for individual, fitness in zip(population, results):
            individual.fitness = fitness
