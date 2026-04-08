from individual import Individual


def traditional(population: list[Individual], offspring: list[Individual]) -> list[Individual]:
    """
    Aplica supervivencia aditiva: combina padres e hijos y conserva los N mejores.
    N corresponde al tamaño de la población original.
    """
    n = len(population)
    return sorted(population + offspring, key=lambda ind: ind.fitness, reverse=True)[:n]


def young_bias(population: list[Individual], offspring: list[Individual]) -> list[Individual]:
    """
    Aplica supervivencia exclusiva: la nueva generación se compone solo de hijos.
    Requiere al menos N hijos, donde N es el tamaño de la población original.
    """
    n = len(population)
    k = len(offspring)

    if k < n:
        raise ValueError(
            f"Supervivencia exclusiva requiere al menos {n} hijos, pero se recibieron {k}."
        )

    return sorted(offspring, key=lambda ind: ind.fitness, reverse=True)[:n]