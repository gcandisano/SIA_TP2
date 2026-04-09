from __future__ import annotations
import math
import random
from individual import Individual, GENES_PER_TRIANGLE


def one_point(parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
    num_triangles = parent1.num_triangles
    cut = random.randint(1, num_triangles - 1) * GENES_PER_TRIANGLE
    genes1, genes2 = parent1.to_genes(), parent2.to_genes()
    child1 = Individual.from_genes(genes1[:cut] + genes2[cut:], num_triangles)
    child2 = Individual.from_genes(genes2[:cut] + genes1[cut:], num_triangles)
    return child1, child2


def two_point(parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
    num_triangles = parent1.num_triangles
    genes1, genes2 = parent1.to_genes(), parent2.to_genes()

    p1, p2 = sorted(random.sample(range(num_triangles), 2))
    cut1 = p1 * GENES_PER_TRIANGLE
    cut2 = p2 * GENES_PER_TRIANGLE

    child1 = Individual.from_genes(genes1[:cut1] + genes2[cut1:cut2] + genes1[cut2:], num_triangles)
    child2 = Individual.from_genes(genes2[:cut1] + genes1[cut1:cut2] + genes2[cut2:], num_triangles)
    return child1, child2


def uniform(parent1: Individual, parent2: Individual, swap_prob: float = 0.5) -> tuple[Individual, Individual]:
    num_triangles = parent1.num_triangles
    genes1, genes2 = parent1.to_genes(), parent2.to_genes()
    child1_genes, child2_genes = genes1[:], genes2[:]

    for triangle_idx in range(num_triangles):
        if random.random() < swap_prob:
            gene_start = triangle_idx * GENES_PER_TRIANGLE
            gene_end = gene_start + GENES_PER_TRIANGLE
            child1_genes[gene_start:gene_end], child2_genes[gene_start:gene_end] = \
                genes2[gene_start:gene_end], genes1[gene_start:gene_end]

    return Individual.from_genes(child1_genes, num_triangles), Individual.from_genes(child2_genes, num_triangles)


def annular(parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
    num_triangles = parent1.num_triangles
    genes1, genes2 = parent1.to_genes(), parent2.to_genes()
    child1_genes, child2_genes = genes1[:], genes2[:]

    start_triangle = random.randint(0, num_triangles - 1)
    segment_length = random.randint(0, math.ceil(num_triangles / 2))

    for offset in range(segment_length):
        triangle_idx = (start_triangle + offset) % num_triangles
        gene_start = triangle_idx * GENES_PER_TRIANGLE
        gene_end = gene_start + GENES_PER_TRIANGLE
        child1_genes[gene_start:gene_end], child2_genes[gene_start:gene_end] = \
            genes2[gene_start:gene_end], genes1[gene_start:gene_end]

    return Individual.from_genes(child1_genes, num_triangles), Individual.from_genes(child2_genes, num_triangles)
