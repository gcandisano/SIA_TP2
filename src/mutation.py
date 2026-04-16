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


def gene(individual: Individual, width: int, height: int, mutation_rate: float = 0.5, **_kwargs: Any) -> Individual:
    """Con probabilidad mutation_rate, muta un único alelo en un triángulo elegido al azar."""
    if random.random() >= mutation_rate:
        return individual # Ya no necesitamos copiar si no mutamos
    
    new_triangles = individual.triangles[:]
    idx = random.randrange(len(new_triangles))
    new_triangles[idx] = new_triangles[idx].copy()
    new_triangles[idx].mutate_one_gene(width, height)
    return Individual(new_triangles)


def multigen(individual: Individual, width: int, height: int, num_genes: int = 3, **_kwargs: Any) -> Individual:
    """Muta exactamente num_genes alelos; cada uno puede caer en cualquier triángulo (con reemplazo)."""
    if num_genes <= 0:
        return individual
    
    new_triangles = individual.triangles[:]
    # Para multigen, necesitamos trackear qué triángulos ya copiamos para no duplicar trabajo
    copied_indices = set()
    
    for _ in range(num_genes):
        idx = random.randrange(len(new_triangles))
        if idx not in copied_indices:
            new_triangles[idx] = new_triangles[idx].copy()
            copied_indices.add(idx)
        new_triangles[idx].mutate_one_gene(width, height)
        
    return Individual(new_triangles)


def non_uniform(individual: Individual, width: int, height: int, mutation_rate: float = 0.02, generation: int = 0, max_generations: int = 1000, b: float = 5.0, **_kwargs: Any) -> Individual:
    """Como mutación de un gen, pero la magnitud del paso decae con la generación."""
    if random.random() >= mutation_rate:
        return individual

    t_ratio = generation / max(max_generations, 1)
    strength = (1.0 - t_ratio) ** max(0.0, b)

    new_triangles = individual.triangles[:]
    idx = random.randrange(len(new_triangles))
    new_triangles[idx] = new_triangles[idx].copy()
    new_triangles[idx].mutate_one_gene(width, height, strength=strength)
    return Individual(new_triangles)
