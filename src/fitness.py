from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw

from individual import Individual


def render(individual: Individual, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for t in individual.triangles:
        vertices = [(t.x1, t.y1), (t.x2, t.y2), (t.x3, t.y3)]
        color = (int(t.r), int(t.g), int(t.b), int(t.a * 255))
        draw.polygon(vertices, fill=color)

    return Image.alpha_composite(canvas, overlay)


def compute_fitness(individual: Individual, target: np.ndarray, width: int, height: int) -> float:
    rendered = np.array(render(individual, width, height).convert("RGB"), dtype=np.float32)
    mse = np.mean(((rendered - target) / 255.0) ** 2)
    return 1.0 - float(mse)


def evaluate_population(population: list[Individual], target: np.ndarray, width: int, height: int) -> None:
    for individual in population:
        individual.fitness = compute_fitness(individual, target, width, height)
