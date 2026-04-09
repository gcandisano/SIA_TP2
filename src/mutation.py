from __future__ import annotations
import random
from typing import Any
from individual import Individual, Triangle


def _mutate_triangle(triangle: Triangle, width: int, height: int) -> Triangle:
    t = triangle.copy()
    t.mutate_positions(width, height)
    t.mutate_color()
    return t


def uniform(individual: Individual, width: int, height: int, mutation_rate: float = 0.02, **_kwargs: Any) -> Individual:
    """Cada triángulo se muta con probabilidad mutation_rate. Utilizamos mutación aleatoria."""
    triangles = [
        Triangle.random(width, height) if random.random() < mutation_rate else t
        for t in individual.triangles
    ]
    return Individual(triangles)


def complete(individual: Individual, width: int, height: int, mutation_rate: float = 0.5, **_kwargs: Any) -> Individual:
    """Con probabilidad mutation_rate, muta todos los triángulos. Utilizamos mutación de un 15% de los alelos."""
    if random.random() >= mutation_rate:
        return individual.copy()
    triangles = [_mutate_triangle(t, width, height) for t in individual.triangles]
    return Individual(triangles)


def gene(individual: Individual, width: int, height: int, **_kwargs: Any) -> Individual:
    raise NotImplementedError


def multigen(individual: Individual, width: int, height: int, num_genes: int = 3, **_kwargs: Any) -> Individual:
    raise NotImplementedError


def non_uniform(individual: Individual, width: int, height: int, mutation_rate: float = 0.02, generation: int = 0, max_generations: int = 1000, **_kwargs: Any) -> Individual:
    raise NotImplementedError
