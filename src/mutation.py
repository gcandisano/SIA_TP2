import random
from individual import Individual, Triangle


def _mutate_triangle(triangle, width, height):
    t = triangle.copy()
    t.mutate_positions(width, height)
    t.mutate_color()
    return t


def uniform(individual, width, height, mutation_rate=0.02, **_):
    """Cada triángulo se muta con probabilidad mutation_rate. Utilizamos mutación aleatoria."""
    triangles = [
        Triangle.random(width, height) if random.random() < mutation_rate else t
        for t in individual.triangles
    ]
    return Individual(triangles)


def complete(individual, width, height, mutation_rate=0.5, **_):
    """Con probabilidad mutation_rate, muta todos los triángulos. Utilizamos mutación de un 15% de los alelos."""
    if random.random() >= mutation_rate:
        return individual.copy()
    triangles = [_mutate_triangle(t, width, height) for t in individual.triangles]
    return Individual(triangles)


def gene(individual, width, height, **_):
    raise NotImplementedError


def multigen(individual, width, height, num_genes=3, **_):
    raise NotImplementedError


def non_uniform(individual, width, height, mutation_rate=0.02, generation=0, max_generations=1000, **_):
    raise NotImplementedError
