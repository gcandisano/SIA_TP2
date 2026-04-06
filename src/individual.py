import random
from copy import deepcopy


GENES_PER_TRIANGLE = 10


class Triangle:
    """
    - x1, y1, x2, y2, x3, y3: coordenadas en píxeles (float).
    - r, g, b: componentes de color en [0.0, 255.0].
    - a: alpha (transparencia) en [0.0, 1.0].
    """

    def __init__(self, x1, y1, x2, y2, x3, y3, r, g, b, a):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    def to_genes(self):
        return [self.x1, self.y1, self.x2, self.y2,
                self.x3, self.y3, self.r, self.g, self.b, self.a]

    @staticmethod
    def from_genes(genes):
        return Triangle(
            genes[0], genes[1],
            genes[2], genes[3],
            genes[4], genes[5],
            genes[6], genes[7], genes[8], genes[9],
        )

    @staticmethod
    def random(width, height):
        return Triangle(
            x1=random.uniform(0, width),
            y1=random.uniform(0, height),
            x2=random.uniform(0, width),
            y2=random.uniform(0, height),
            x3=random.uniform(0, width),
            y3=random.uniform(0, height),
            r=random.uniform(0, 255),
            g=random.uniform(0, 255),
            b=random.uniform(0, 255),
            a=random.uniform(0.1, 1.0),
        )

    def copy(self):
        return Triangle(self.x1, self.y1, self.x2, self.y2,
                        self.x3, self.y3, self.r, self.g, self.b, self.a)

    def mutate_positions(self, width, height):
        self.x1 = self.x1 * random.uniform(0.85, 1.15) % width
        self.y1 = self.y1 * random.uniform(0.85, 1.15) % height
        self.x2 = self.x2 * random.uniform(0.85, 1.15) % width
        self.y2 = self.y2 * random.uniform(0.85, 1.15) % height
        self.x3 = self.x3 * random.uniform(0.85, 1.15) % width
        self.y3 = self.y3 * random.uniform(0.85, 1.15) % height

    def mutate_color(self):
        self.r = self.r * random.uniform(0.85, 1.15) % 256
        self.g = self.g * random.uniform(0.85, 1.15) % 256
        self.b = self.b * random.uniform(0.85, 1.15) % 256
        self.a = min(1.0, self.a * random.uniform(0.85, 1.15))

    def clamp(self, width, height):
        """Devuelve un nuevo Triangle con todos los genes dentro de rango."""
        return Triangle(
            x1=max(0.0, min(self.x1, width)),
            y1=max(0.0, min(self.y1, height)),
            x2=max(0.0, min(self.x2, width)),
            y2=max(0.0, min(self.y2, height)),
            x3=max(0.0, min(self.x3, width)),
            y3=max(0.0, min(self.y3, height)),
            r=max(0.0, min(self.r, 255.0)),
            g=max(0.0, min(self.g, 255.0)),
            b=max(0.0, min(self.b, 255.0)),
            a=max(0.0, min(self.a, 1.0)),
        )


class Individual:
    """
    N triángulos pintados en orden sobre un canvas blanco.
    """

    def __init__(self, triangles, fitness=0.0):
        self.triangles = triangles
        self.fitness = fitness

    @property
    def num_triangles(self):
        return len(self.triangles)

    def to_genes(self):
        genes = []
        for t in self.triangles:
            genes.extend(t.to_genes())
        return genes

    @staticmethod
    def from_genes(genes, num_triangles):
        p = GENES_PER_TRIANGLE
        triangles = [
            Triangle.from_genes(genes[i * p: (i + 1) * p])
            for i in range(num_triangles)
        ]
        return Individual(triangles)

    @staticmethod
    def random(num_triangles, width, height):
        return Individual(
            [Triangle.random(width, height) for _ in range(num_triangles)]
        )

    def copy(self):
        return deepcopy(self)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __le__(self, other):
        return self.fitness <= other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __ge__(self, other):
        return self.fitness >= other.fitness

    def __repr__(self):
        return f"Individual(triangles={self.num_triangles}, fitness={self.fitness:.6f})"
