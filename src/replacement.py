def traditional(population, offspring):
    """
    Selecciona N individuos del conjunto unificado de padres e hijos.
    N = tamaño de la población original.
    """
    n = len(population)
    return sorted(population + offspring, reverse=True)[:n]


def young_bias(population, offspring):
    """
    K > N : selecciona N de los K hijos únicamente.
    K <= N: toma los K hijos + (N-K) mejores padres.
    """
    n = len(population)
    k = len(offspring)

    if k > n:
        return sorted(offspring, reverse=True)[:n]

    best_parents = sorted(population, reverse=True)[: n - k]
    return offspring + best_parents
