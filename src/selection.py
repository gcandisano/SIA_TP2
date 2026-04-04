from individual import Individual


def elite(population, k):
    return sorted(population, reverse=True)[:k]


def roulette(population, k):
    raise NotImplementedError


def universal(population, k):
    raise NotImplementedError


def boltzmann(population, k, temperature):
    raise NotImplementedError


def tournament_deterministic(population, k, tournament_size):
    raise NotImplementedError


def tournament_probabilistic(population, k, threshold):
    raise NotImplementedError


def ranking(population, k):
    raise NotImplementedError
