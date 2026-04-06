import random
from individual import Individual, GENES_PER_TRIANGLE


def one_point(parent1, parent2):
    n = parent1.num_triangles
    cut = random.randint(1, n - 1) * GENES_PER_TRIANGLE
    g1, g2 = parent1.to_genes(), parent2.to_genes()
    child1 = Individual.from_genes(g1[:cut] + g2[cut:], n)
    child2 = Individual.from_genes(g2[:cut] + g1[cut:], n)
    return child1, child2


def two_point(parent1, parent2):
    raise NotImplementedError


def uniform(parent1, parent2, swap_prob=0.5):
    raise NotImplementedError


def annular(parent1, parent2):
    raise NotImplementedError
