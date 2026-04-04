from individual import Individual


def initialize(size, num_triangles, width, height):
    return [Individual.random(num_triangles, width, height) for _ in range(size)]
