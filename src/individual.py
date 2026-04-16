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
        return [
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            self.x3,
            self.y3,
            self.r,
            self.g,
            self.b,
            self.a,
        ]

    @staticmethod
    def from_genes(genes: list[float]) -> Triangle:
        return Triangle(
            genes[0],
            genes[1],
            genes[2],
            genes[3],
            genes[4],
            genes[5],
            genes[6],
            genes[7],
            genes[8],
            genes[9],
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
        return Triangle(
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            self.x3,
            self.y3,
            self.r,
            self.g,
            self.b,
            self.a,
        )

    def mutate_positions(self, width: int, height: int) -> None:
        dx = width * 0.1
        dy = height * 0.1
        self.x1 = max(0, min(width, self.x1 + random.uniform(-dx, dx)))
        self.y1 = max(0, min(height, self.y1 + random.uniform(-dy, dy)))
        self.x2 = max(0, min(width, self.x2 + random.uniform(-dx, dx)))
        self.y2 = max(0, min(height, self.y2 + random.uniform(-dy, dy)))
        self.x3 = max(0, min(width, self.x3 + random.uniform(-dx, dx)))
        self.y3 = max(0, min(height, self.y3 + random.uniform(-dy, dy)))

    def mutate_one_gene(self, width: int, height: int, strength: float = 1.0) -> None:
        """Muta exactamente un alelo (un gen de los 10 del triángulo) con cambios aditivos y clamping."""
        strength = max(0.0, strength)
        gene_index: int = random.randint(0, 9)

        # Ajustamos el rango de mutación según el tipo de gen
        if gene_index < 6:  # Coordenadas (x, y)
            if gene_index % 2 == 0:  # X-coordinates (0, 2, 4)
                delta = random.uniform(-width * 0.1, width * 0.1) * strength
                if gene_index == 0:
                    self.x1 = max(0, min(width, self.x1 + delta))
                elif gene_index == 2:
                    self.x2 = max(0, min(width, self.x2 + delta))
                elif gene_index == 4:
                    self.x3 = max(0, min(width, self.x3 + delta))
            else:  # Y-coordinates (1, 3, 5)
                delta = random.uniform(-height * 0.1, height * 0.1) * strength
                if gene_index == 1:
                    self.y1 = max(0, min(height, self.y1 + delta))
                elif gene_index == 3:
                    self.y2 = max(0, min(height, self.y2 + delta))
                elif gene_index == 5:
                    self.y3 = max(0, min(height, self.y3 + delta))
        elif gene_index < 9:  # Colores (r, g, b)
            delta = random.uniform(-40, 40) * strength
            if gene_index == 6:
                self.r = max(0, min(255, self.r + delta))
            elif gene_index == 7:
                self.g = max(0, min(255, self.g + delta))
            elif gene_index == 8:
                self.b = max(0, min(255, self.b + delta))
        else:  # Alpha (a)
            delta = random.uniform(-0.1, 0.1) * strength
            self.a = max(0.05, min(1.0, self.a + delta))

    def mutate_color(self) -> None:
        self.r = max(0, min(255, self.r + random.uniform(-30, 30)))
        self.g = max(0, min(255, self.g + random.uniform(-30, 30)))
        self.b = max(0, min(255, self.b + random.uniform(-30, 30)))
        self.a = max(0.05, min(1.0, self.a + random.uniform(-0.1, 0.1)))

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

    def __init__(
        self,
        triangles: list[Triangle],
        fitness: float = 0.0,
        genes: list[float] | None = None,
    ):
        self.triangles = triangles
        self.fitness = fitness
        self._genes = genes  # Pass existing genes to pre-populate cache

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
            Triangle.from_genes(genes[i * p : (i + 1) * p])
            for i in range(num_triangles)
        ]
        # Pre-populate the cache since we already have the genes
        return Individual(triangles, genes=genes)

    @staticmethod
    def random(num_triangles: int, width: int, height: int) -> Individual:
        return Individual(
            [Triangle.random(width, height) for _ in range(num_triangles)]
        )

    def copy(self) -> Individual:
        # Shallow copy of the list is enough if we treat Triangle objects as semi-immutable
        # when they are shared between individuals.
        return Individual(self.triangles[:], fitness=self.fitness)

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
