import argparse
import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, "src")

import crossover as cx
import mutation as mut
import replacement as rep
import selection as sel
from fitness import evaluate_population, render
from population import initialize


def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


SELECTION_METHODS = {
    "elite": sel.elite,
    "roulette": sel.roulette,
    "universal": sel.universal,
    "boltzmann": sel.boltzmann,
    "tournament_det": sel.tournament_deterministic,
    "tournament_prob": sel.tournament_probabilistic,
    "ranking": sel.ranking,
}

CROSSOVER_METHODS = {
    "one_point": cx.one_point,
    "two_point": cx.two_point,
    "uniform": cx.uniform,
    "annular": cx.annular,
}

MUTATION_METHODS = {
    "gene": mut.gene,
    "multigen": mut.multigen,
    "uniform": mut.uniform,
    "complete": mut.complete,
    "non_uniform": mut.non_uniform,
}

REPLACEMENT_METHODS = {
    "traditional": rep.traditional,
    "young_bias": rep.young_bias,
}


def run(config_path="config.json"):
    cfg = load_config(config_path)

    target_img = Image.open(cfg["image_path"]).convert("RGB")
    width, height = target_img.size
    target = np.array(target_img, dtype=np.float32)

    select = SELECTION_METHODS[cfg["selection"]["method"]]
    cross = CROSSOVER_METHODS[cfg["crossover"]["method"]]
    mutate = MUTATION_METHODS[cfg["mutation"]["method"]]
    replace = REPLACEMENT_METHODS[cfg["replacement"]["method"]]

    population = initialize(cfg["population_size"], cfg["num_triangles"], width, height)
    evaluate_population(population, target, width, height)

    best = max(population)
    best_fitness_per_generation = [best.fitness]
    print(f"Gen 0 | best fitness: {best.fitness:.6f}")

    stagnation_n = cfg.get("stagnation_generations")
    stagnation_eps = cfg.get("stagnation_epsilon", 1e-6)
    use_stagnation = stagnation_n is not None and stagnation_n > 0
    best_ever = best.fitness
    gens_without_gain = 0

    for generation in range(1, cfg["max_generations"] + 1):
        parents = select(population, cfg["selection"]["k"])

        random.shuffle(parents)
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            child1, child2 = cross(parents[i], parents[i + 1])
            offspring.append(child1)
            offspring.append(child2)

        mut_params = cfg["mutation"]
        offspring = [
            mutate(
                ind,
                width,
                height,
                generation=generation,
                max_generations=cfg["max_generations"],
                **mut_params,
            )
            for ind in offspring
        ]

        evaluate_population(offspring, target, width, height)

        population = replace(population, offspring)

        best = max(population)
        best_fitness_per_generation.append(best.fitness)
        print(f"Gen {generation} | best fitness: {best.fitness:.6f}")

        if best.fitness >= cfg.get("target_fitness", 1.0):
            print("Criterio de parada: fitness objetivo alcanzado.")
            break

        if use_stagnation:
            if best.fitness > best_ever + stagnation_eps:
                best_ever = best.fitness
                gens_without_gain = 0
            else:
                gens_without_gain += 1
                if best.fitness > best_ever:
                    best_ever = best.fitness
                if gens_without_gain >= stagnation_n:
                    print(
                        "Criterio de parada: estancamiento "
                        f"({stagnation_n} generaciones sin mejora > {stagnation_eps})."
                    )
                    break

    os.makedirs("output", exist_ok=True)
    plot_fitness(best_fitness_per_generation)
    render(best, width, height).convert("RGB").save("output/result.png")
    print("Imagen guardada en output/result.png")


def plot_fitness(best_fitness_per_generation: list[float]):
    fitness_plot_path = "output/best_fitness.png"
    gens = range(len(best_fitness_per_generation))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, best_fitness_per_generation, color="C0")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Mejor fitness")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fitness_plot_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compresor de Imágenes mediante Algoritmos Genéticos"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.json",
        help="Ruta al archivo de configuración JSON (por defecto: config.json)",
    )

    args = parser.parse_args()
    run(args.config)
