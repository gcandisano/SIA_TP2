import math
import random


def elite(population, k):
    return sorted(population, reverse=True)[:k]


def roulette(population, k):
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


def universal(population, k):
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


def boltzmann(population, k, temperature=1.0):
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


def tournament_deterministic(population, k, tournament_size=3):
    selected = []
    for _ in range(k):
        contestants = random.sample(population, min(tournament_size, len(population)))
        selected.append(max(contestants))
    return selected


def tournament_probabilistic(population, k, threshold=0.75):
    selected = []
    for _ in range(k):
        a, b = random.sample(population, 2)
        best, worst = (a, b) if a >= b else (b, a)

        r = random.uniform(0, 1)
        selected.append(best if r < threshold else worst)
    return selected


def ranking(population, k):
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
