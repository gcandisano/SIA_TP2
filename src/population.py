from __future__ import annotations
from individual import Individual


def initialize(size: int, num_triangles: int, width: int, height: int) -> list[Individual]:
    return [Individual.random(num_triangles, width, height) for _ in range(size)]
