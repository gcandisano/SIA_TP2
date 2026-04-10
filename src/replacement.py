from __future__ import annotations
from individual import Individual


def traditional(population: list[Individual], offspring: list[Individual]) -> list[Individual]:
    """
    Aplica supervivencia aditiva: combina padres e hijos y conserva los N mejores.
    N corresponde al tamaño de la población original.
    """
    n = len(population)
    return sorted(population + offspring, key=lambda ind: ind.fitness, reverse=True)[:n]


def young_bias(population: list[Individual], offspring: list[Individual]) -> list[Individual]:
    """
    Aplica supervivencia exclusiva:
    - K > N: selecciona los N mejores de los K hijos.
    - K ≤ N: toma los K hijos + los (N-K) mejores de la generación actual.
    """
    n = len(population)
    k = len(offspring)

    if k > n:
        return sorted(offspring, key=lambda ind: ind.fitness, reverse=True)[:n]

    remaining = n - k
    best_from_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)[:remaining]
    return offspring + best_from_population