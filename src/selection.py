from __future__ import annotations
import math
import random
from individual import Individual


def elite(population: list[Individual], k: int) -> list[Individual]:
    return sorted(population, reverse=True)[:k]


def roulette(population: list[Individual], k: int) -> list[Individual]:
    total = sum(ind.fitness for ind in population)

    p = [ind.fitness / total for ind in population]

    q = []
    cumulative = 0.0
    for pi in p:
        cumulative += pi
        q.append(cumulative)

    selected = []
    for _ in range(k):
        r = random.uniform(0, 1)
        for i, qi in enumerate(q):
            if r <= qi:
                selected.append(population[i])
                break

    return selected


def universal(population: list[Individual], k: int) -> list[Individual]:
    total = sum(ind.fitness for ind in population)

    p = [ind.fitness / total for ind in population]

    q = []
    cumulative = 0.0
    for pi in p:
        cumulative += pi
        q.append(cumulative)

    r = random.uniform(0, 1)
    pointers = [(r + j) / k for j in range(k)]

    selected = []
    for rj in pointers:
        for i, qi in enumerate(q):
            if rj <= qi:
                selected.append(population[i])
                break

    return selected


def boltzmann(population: list[Individual], k: int, temperature: float = 1.0) -> list[Individual]:
    exp_vals = [math.exp(ind.fitness / temperature) for ind in population]
    avg_exp = sum(exp_vals) / len(exp_vals)

    # Pseudo-aptitud: ExpVal(i) = e^(f(i)/T) / <e^(f(x)/T)>
    pseudo_fitness = [ev / avg_exp for ev in exp_vals]

    total = sum(pseudo_fitness)
    q = []
    cumulative = 0.0
    for pf in pseudo_fitness:
        cumulative += pf / total
        q.append(cumulative)

    selected = []
    for _ in range(k):
        r = random.uniform(0, 1)
        for i, qi in enumerate(q):
            if r <= qi:
                selected.append(population[i])
                break

    return selected


def tournament_deterministic(population: list[Individual], k: int, tournament_size: int = 3) -> list[Individual]:
    selected = []
    for _ in range(k):
        contestants = random.sample(population, min(tournament_size, len(population)))
        selected.append(max(contestants))
    return selected


def tournament_probabilistic(population: list[Individual], k: int, threshold: float = 0.75) -> list[Individual]:
    selected = []
    for _ in range(k):
        a, b = random.sample(population, 2)
        best, worst = (a, b) if a >= b else (b, a)

        r = random.uniform(0, 1)
        selected.append(best if r < threshold else worst)
    return selected


def ranking(population: list[Individual], k: int) -> list[Individual]:
    n = len(population)

    sorted_pop = sorted(population, reverse=True)
    pseudo_fitness = [(n - rank) / n for rank in range(1, n + 1)]

    total = sum(pseudo_fitness)
    q = []
    cumulative = 0.0
    for pf in pseudo_fitness:
        cumulative += pf / total
        q.append(cumulative)

    selected = []
    for _ in range(k):
        r = random.uniform(0, 1)
        for i, qi in enumerate(q):
            if r <= qi:
                selected.append(sorted_pop[i])
                break

    return selected
