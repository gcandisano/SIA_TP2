from __future__ import annotations

import random

GENES_PER_TRIANGLE: int = 10


class Triangle:
    """
    - x1, y1, x2, y2, x3, y3: coordenadas en píxeles (float).
    - r, g, b: componentes de color en [0.0, 255.0].
    - a: alpha (transparencia) en [0.0, 1.0].
    """

    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        x3: float,
        y3: float,
        r: float,
        g: float,
        b: float,
        a: float,
    ):
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

    def to_genes(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2,
                self.x3, self.y3, self.r, self.g, self.b, self.a]

    @staticmethod
    def from_genes(genes: list[float]) -> Triangle:
        return Triangle(
            genes[0], genes[1],
            genes[2], genes[3],
            genes[4], genes[5],
            genes[6], genes[7], genes[8], genes[9],
        )

    @staticmethod
    def random(width: int, height: int) -> Triangle:
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

    def copy(self) -> Triangle:
        return Triangle(self.x1, self.y1, self.x2, self.y2,
                        self.x3, self.y3, self.r, self.g, self.b, self.a)

    def mutate_positions(self, width: int, height: int) -> None:
        self.x1 = self.x1 * random.uniform(0.85, 1.15) % width
        self.y1 = self.y1 * random.uniform(0.85, 1.15) % height
        self.x2 = self.x2 * random.uniform(0.85, 1.15) % width
        self.y2 = self.y2 * random.uniform(0.85, 1.15) % height
        self.x3 = self.x3 * random.uniform(0.85, 1.15) % width
        self.y3 = self.y3 * random.uniform(0.85, 1.15) % height

    def mutate_one_gene(self, width: int, height: int, strength: float = 1.0) -> None:
        """Muta exactamente un alelo (un gen de los 10 del triángulo)"""
        strength = max(0.0, strength)
        gene_index: int = random.randint(0, 9)
        span = 0.15 * strength
        lo, hi = max(0.0, 1.0 - span), 1.0 + span
        factor = random.uniform(lo, hi)

        if gene_index == 0:
            self.x1 = self.x1 * factor % width
            return
        if gene_index == 1:
            self.y1 = self.y1 * factor % height
            return
        if gene_index == 2:
            self.x2 = self.x2 * factor % width
            return
        if gene_index == 3:
            self.y2 = self.y2 * factor % height
            return
        if gene_index == 4:
            self.x3 = self.x3 * factor % width
            return
        if gene_index == 5:
            self.y3 = self.y3 * factor % height
            return
        if gene_index == 6:
            self.r = self.r * factor % 256
            return
        if gene_index == 7:
            self.g = self.g * factor % 256
            return
        if gene_index == 8:
            self.b = self.b * factor % 256
            return
            
        self.a = max(0.1, min(1.0, self.a * factor))

    def mutate_color(self) -> None:
        self.r = self.r * random.uniform(0.85, 1.15) % 256
        self.g = self.g * random.uniform(0.85, 1.15) % 256
        self.b = self.b * random.uniform(0.85, 1.15) % 256
        self.a = min(1.0, self.a * random.uniform(0.85, 1.15))

    def clamp(self, width: int, height: int) -> Triangle:
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

    def __init__(self, triangles: list[Triangle], fitness: float = 0.0):
        self.triangles = triangles
        self.fitness = fitness
        self._genes: list[float] | None = None

    @property
    def num_triangles(self) -> int:
        return len(self.triangles)

    def to_genes(self) -> list[float]:
        if self._genes is None:
            self._genes = [gene for t in self.triangles for gene in t.to_genes()]
        return self._genes

    @staticmethod
    def from_genes(genes: list[float], num_triangles: int) -> Individual:
        p = GENES_PER_TRIANGLE
        triangles = [
            Triangle.from_genes(genes[i * p: (i + 1) * p])
            for i in range(num_triangles)
        ]
        return Individual(triangles)

    @staticmethod
    def random(num_triangles: int, width: int, height: int) -> Individual:
        return Individual(
            [Triangle.random(width, height) for _ in range(num_triangles)]
        )

    def copy(self) -> Individual:
        return Individual([t.copy() for t in self.triangles], fitness=self.fitness)

    def __lt__(self, other: Individual) -> bool:
        return self.fitness < other.fitness

    def __le__(self, other: Individual) -> bool:
        return self.fitness <= other.fitness

    def __gt__(self, other: Individual) -> bool:
        return self.fitness > other.fitness

    def __ge__(self, other: Individual) -> bool:
        return self.fitness >= other.fitness

    def __repr__(self) -> str:
        return f"Individual(triangles={self.num_triangles}, fitness={self.fitness:.6f})"
