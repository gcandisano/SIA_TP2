import argparse
import csv
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
from individual import Individual
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

    selection_params = {k: v for k, v in cfg["selection"].items() if k not in ["method", "k"]}
    crossover_params = {k: v for k, v in cfg["crossover"].items() if k != "method"}
    
    select = SELECTION_METHODS[cfg["selection"]["method"]]
    cross = CROSSOVER_METHODS[cfg["crossover"]["method"]]
    mutate = MUTATION_METHODS[cfg["mutation"]["method"]]
    replace = REPLACEMENT_METHODS[cfg["replacement"]["method"]]

    population = initialize(cfg["population_size"], cfg["num_triangles"], width, height)
    evaluate_population(population, target, width, height)

    # Pre-calculamos el vector de normalización para la diversidad
    ranges = np.array([width, height, width, height, width, height, 255.0, 255.0, 255.0, 1.0])
    norm_vector = np.tile(ranges, cfg["num_triangles"])

    best = max(population)
    fitness_values = [ind.fitness for ind in population]
    best_fitness_per_generation = [best.fitness]
    avg_fitness_per_generation = [float(np.mean(fitness_values))]
    diversity_per_generation = [_compute_diversity(population, norm_vector)]
    print(f"Gen 0 | best fitness: {best.fitness:.6f}")

    stagnation_n = cfg.get("stagnation_generations")
    stagnation_eps = cfg.get("stagnation_epsilon", 1e-6)
    use_stagnation = stagnation_n is not None and stagnation_n > 0
    best_ever = best.fitness
    gens_without_gain = 0

    try:
        for generation in range(1, cfg["max_generations"] + 1):
            parents = select(population, cfg["selection"]["k"], **selection_params)

            random.shuffle(parents)
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                child1, child2 = cross(parents[i], parents[i + 1], **crossover_params)
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
            fitness_values = [ind.fitness for ind in population]
            best_fitness_per_generation.append(best.fitness)
            avg_fitness_per_generation.append(float(np.mean(fitness_values)))
            if generation % 10 == 0:
                diversity_per_generation.append(_compute_diversity(population, norm_vector))
            else:
                diversity_per_generation.append(diversity_per_generation[-1])
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
    except KeyboardInterrupt:
        print("\nEjecución interrumpida por el usuario. Guardando estado actual...")

    os.makedirs("output", exist_ok=True)
    plot_fitness(best_fitness_per_generation, avg_fitness_per_generation)
    plot_diversity(diversity_per_generation)
    save_metrics_csv(best_fitness_per_generation, avg_fitness_per_generation, diversity_per_generation)
    render(best, width, height).convert("RGB").save("output/result.png")
    print("Imagen guardada en output/result.png")
    save_triangles(best)

def save_triangles(individual: Individual) -> None:
    triangles_path = "output/triangles.json"
    data = {
        "num_triangles": individual.num_triangles,
        "fitness": individual.fitness,
        "triangles": [
            {
                "index": i,
                "vertices": {
                    "x1": round(t.x1, 4), "y1": round(t.y1, 4),
                    "x2": round(t.x2, 4), "y2": round(t.y2, 4),
                    "x3": round(t.x3, 4), "y3": round(t.y3, 4),
                },
                "color": {
                    "r": round(t.r, 4),
                    "g": round(t.g, 4),
                    "b": round(t.b, 4),
                    "a": round(t.a, 4),
                    "hex": "#{:02x}{:02x}{:02x}".format(int(t.r), int(t.g), int(t.b)),
                },
            }
            for i, t in enumerate(individual.triangles)
        ],
    }
    with open(triangles_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Triángulos guardados en {triangles_path}")


def _compute_diversity(population: list[Individual], norm_vector: np.ndarray) -> float:
    """Diversidad genética normalizada: desviación estándar media de los genes normalizados [0,1]."""
    if not population:
        return 0.0
    
    genes = np.array([ind.to_genes() for ind in population])
    normalized_genes = genes / norm_vector
    return float(np.mean(np.std(normalized_genes, axis=0)))


def plot_fitness(
    best_fitness_per_generation: list[float],
    avg_fitness_per_generation: list[float],
):
    fitness_plot_path = "output/best_fitness.png"
    gens = range(len(best_fitness_per_generation))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, best_fitness_per_generation, color="C0", label="Mejor fitness")
    ax.plot(gens, avg_fitness_per_generation, color="C1", label="Fitness promedio", alpha=0.7)
    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fitness_plot_path, dpi=150)
    plt.close(fig)


def plot_diversity(diversity_per_generation: list[float]):
    diversity_plot_path = "output/diversity.png"
    gens = range(len(diversity_per_generation))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, diversity_per_generation, color="C2")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Diversidad genética (σ media de genes)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(diversity_plot_path, dpi=150)
    plt.close(fig)


def save_metrics_csv(
    best_fitness: list[float],
    avg_fitness: list[float],
    diversity: list[float],
):
    csv_path = "output/metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "avg_fitness", "diversity"])
        for gen in range(len(best_fitness)):
            writer.writerow([
                gen,
                f"{best_fitness[gen]:.8f}",
                f"{avg_fitness[gen]:.8f}",
                f"{diversity[gen]:.8f}",
            ])
    print(f"Métricas guardadas en {csv_path}")


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
