import argparse
import copy
import csv
import inspect
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Optional

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


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def expand_runs(loaded: dict) -> list[tuple[str, dict]]:
    if "runs" not in loaded:
        return [("run", loaded)]

    runs = loaded["runs"]
    if not isinstance(runs, list):
        raise ValueError("El campo 'runs' debe ser una lista de configuraciones.")
    if not runs:
        raise ValueError("El campo 'runs' no puede ser una lista vacía.")

    base_cfg = {k: v for k, v in loaded.items() if k != "runs"}
    expanded: list[tuple[str, dict]] = []
    for idx, run_override in enumerate(runs, start=1):
        if not isinstance(run_override, dict):
            raise ValueError(
                f"Cada elemento de 'runs' debe ser un objeto. Error en índice {idx - 1}."
            )
        override_cfg = copy.deepcopy(run_override)
        label = override_cfg.pop("label", override_cfg.pop("name", f"run_{idx}"))
        expanded.append((str(label), deep_merge(base_cfg, override_cfg)))
    return expanded


def slugify_label(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower())
    slug = slug.strip("_")
    return slug or "run"


def filter_supported_kwargs(func, kwargs: dict) -> dict:
    signature = inspect.signature(func)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


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


@dataclass
class RunResult:
    best: Individual
    best_fitness_per_generation: list[float]
    avg_fitness_per_generation: list[float]
    diversity_per_generation: list[float]
    width: int
    height: int


def run_ga(cfg: dict, label: Optional[str] = None) -> RunResult:
    prefix = f"[{label}] " if label else ""

    if "seed" in cfg:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])

    target_img = Image.open(cfg["image_path"]).convert("RGB")
    width, height = target_img.size
    target = np.array(target_img, dtype=np.float32)

    selection_params = {k: v for k, v in cfg["selection"].items() if k not in ["method", "k"]}
    crossover_params = {k: v for k, v in cfg["crossover"].items() if k != "method"}
    
    select = SELECTION_METHODS[cfg["selection"]["method"]]
    cross = CROSSOVER_METHODS[cfg["crossover"]["method"]]
    mutate = MUTATION_METHODS[cfg["mutation"]["method"]]
    replace = REPLACEMENT_METHODS[cfg["replacement"]["method"]]
    selection_params = filter_supported_kwargs(select, selection_params)
    crossover_params = filter_supported_kwargs(cross, crossover_params)

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
    print(f"{prefix}Gen 0 | best fitness: {best.fitness:.6f}")

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
                    **filter_supported_kwargs(
                        mutate,
                        {
                            **mut_params,
                            "generation": generation,
                            "max_generations": cfg["max_generations"],
                        },
                    ),
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
            print(f"{prefix}Gen {generation} | best fitness: {best.fitness:.6f}")

            if best.fitness >= cfg.get("target_fitness", 1.0):
                print(f"{prefix}Criterio de parada: fitness objetivo alcanzado.")
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
                            f"{prefix}Criterio de parada: estancamiento "
                            f"({stagnation_n} generaciones sin mejora > {stagnation_eps})."
                        )
                        break
    except KeyboardInterrupt:
        print(f"\n{prefix}Ejecución interrumpida por el usuario. Guardando estado actual...")

    return RunResult(
        best=best,
        best_fitness_per_generation=best_fitness_per_generation,
        avg_fitness_per_generation=avg_fitness_per_generation,
        diversity_per_generation=diversity_per_generation,
        width=width,
        height=height,
    )


def run(config_path="config.json"):
    loaded_cfg = load_config(config_path)
    run_configs = expand_runs(loaded_cfg)
    os.makedirs("output", exist_ok=True)

    if len(run_configs) == 1:
        _, cfg = run_configs[0]
        result = run_ga(cfg)
        plot_fitness(result.best_fitness_per_generation, result.avg_fitness_per_generation)
        plot_diversity(result.diversity_per_generation)
        save_metrics_csv(
            result.best_fitness_per_generation,
            result.avg_fitness_per_generation,
            result.diversity_per_generation,
        )
        render(result.best, result.width, result.height).convert("RGB").save("output/result.png")
        print("Imagen guardada en output/result.png")
        save_triangles(result.best)
        return

    best_histories: dict[str, list[float]] = {}
    diversity_histories: dict[str, list[float]] = {}
    used_slugs: set[str] = set()

    for label, cfg in run_configs:
        print(f"\n=== Ejecutando configuración: {label} ===")
        result = run_ga(cfg, label=label)
        best_histories[label] = result.best_fitness_per_generation
        diversity_histories[label] = result.diversity_per_generation

        base_slug = slugify_label(label)
        slug = base_slug
        suffix = 2
        while slug in used_slugs:
            slug = f"{base_slug}_{suffix}"
            suffix += 1
        used_slugs.add(slug)

        result_path = f"output/result_{slug}.png"
        metrics_path = f"output/metrics_{slug}.csv"
        triangles_path = f"output/triangles_{slug}.json"

        render(result.best, result.width, result.height).convert("RGB").save(result_path)
        print(f"Imagen guardada en {result_path}")
        save_triangles(result.best, triangles_path)
        save_metrics_csv(
            result.best_fitness_per_generation,
            result.avg_fitness_per_generation,
            result.diversity_per_generation,
            metrics_path,
        )

    plot_fitness_comparison(best_histories)
    plot_diversity_comparison(diversity_histories)

def save_triangles(individual: Individual, triangles_path: str = "output/triangles.json") -> None:
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
    fitness_plot_path: str = "output/best_fitness.png",
):
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


def plot_fitness_comparison(
    best_histories: dict[str, list[float]],
    fitness_plot_path: str = "output/best_fitness.png",
):
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, (label, history) in enumerate(best_histories.items()):
        ax.plot(range(len(history)), history, color=f"C{idx % 10}", label=label)
    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fitness_plot_path, dpi=150)
    plt.close(fig)


def plot_diversity(
    diversity_per_generation: list[float],
    diversity_plot_path: str = "output/diversity.png",
):
    gens = range(len(diversity_per_generation))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, diversity_per_generation, color="C2")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Diversidad genética (σ media de genes)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(diversity_plot_path, dpi=150)
    plt.close(fig)


def plot_diversity_comparison(
    diversity_histories: dict[str, list[float]],
    diversity_plot_path: str = "output/diversity.png",
):
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, (label, history) in enumerate(diversity_histories.items()):
        ax.plot(range(len(history)), history, color=f"C{idx % 10}", label=label)
    ax.set_xlabel("Generación")
    ax.set_ylabel("Diversidad genética (σ media de genes)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(diversity_plot_path, dpi=150)
    plt.close(fig)


def save_metrics_csv(
    best_fitness: list[float],
    avg_fitness: list[float],
    diversity: list[float],
    csv_path: str = "output/metrics.csv",
):
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
