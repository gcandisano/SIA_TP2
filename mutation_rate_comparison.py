"""
mutation_rate_comparison.py — Comparación de probabilidades de mutación (0.01 vs 0.05).

Selección: elite  |  Cruza: dos puntos  |  Reemplazo: tradicional
Métodos de mutación: uniforme y por gen

Genera 2 gráficos de líneas (uno por método de mutación), cada uno con las curvas
de mejor fitness por generación para mutation_rate=0.01 y mutation_rate=0.05.
Cada curva es el promedio de 3 corridas.

Ejecutar:  python mutation_rate_comparison.py
"""

import inspect
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, "src")

import crossover as cx
import mutation as mut
import replacement as rep
import selection as sel
from fitness import evaluate_population
from population import initialize


# ── Parámetros globales ───────────────────────────────────────────────────────

NUM_RUNS        = 3
BASE_SEED       = 42
MAX_GENERATIONS = 500
POPULATION_SIZE = 50
NUM_TRIANGLES   = 50

IMAGE_PATH = "assets/chrome.png"
IMAGE_NAME = "chrome"

MUTATION_RATES = [0.01, 0.5]

CONFIGS = {
    "uniform": {
        "selection":   {"method": "roulette", "k": POPULATION_SIZE},
        "crossover":   {"method": "uniform", "swap_prob": 0.5},
        "replacement": {"method": "traditional"},
    },
    "gene": {
        "selection":   {"method": "roulette", "k": POPULATION_SIZE},
        "crossover":   {"method": "uniform", "swap_prob": 0.5},
        "replacement": {"method": "traditional"},
    },
}

MUTATION_LABELS = {
    "uniform": "Mutación uniforme",
    "gene":    "Mutación por gen",
}

RATE_COLORS = {
    0.01: "#4C72B0",
    0.5: "#C44E52",
}

# ── Mapas de métodos ──────────────────────────────────────────────────────────

SELECTION_METHODS = {
    "elite":           sel.elite,
    "roulette":        sel.roulette,
    "universal":       sel.universal,
    "boltzmann":       sel.boltzmann,
    "tournament_det":  sel.tournament_deterministic,
    "tournament_prob": sel.tournament_probabilistic,
    "ranking":         sel.ranking,
}

CROSSOVER_METHODS = {
    "one_point": cx.one_point,
    "two_point": cx.two_point,
    "uniform":   cx.uniform,
    "annular":   cx.annular,
}

MUTATION_METHODS = {
    "gene":        mut.gene,
    "multigen":    mut.multigen,
    "uniform":     mut.uniform,
    "complete":    mut.complete,
    "non_uniform": mut.non_uniform,
}

REPLACEMENT_METHODS = {
    "traditional": rep.traditional,
    "young_bias":  rep.young_bias,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filter_kwargs(func, kwargs: dict) -> dict:
    sig = inspect.signature(func)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def run_ga(cfg: dict, mutation_method: str, mutation_rate: float,
           target: np.ndarray, width: int, height: int,
           seed: int | None = None) -> list[float]:
    """Ejecuta el AG y devuelve la lista de mejor fitness por generación."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    select  = SELECTION_METHODS[cfg["selection"]["method"]]
    cross   = CROSSOVER_METHODS[cfg["crossover"]["method"]]
    mutate  = MUTATION_METHODS[mutation_method]
    replace = REPLACEMENT_METHODS[cfg["replacement"]["method"]]

    sel_params = _filter_kwargs(select, {k: v for k, v in cfg["selection"].items() if k not in ("method", "k")})
    cx_params  = _filter_kwargs(cross,  {k: v for k, v in cfg["crossover"].items() if k != "method"})
    mut_params = {"mutation_rate": mutation_rate}

    population = initialize(POPULATION_SIZE, NUM_TRIANGLES, width, height)
    evaluate_population(population, target, width, height)

    best_per_gen = [max(population).fitness]

    for gen in range(1, MAX_GENERATIONS + 1):
        parents = select(population, cfg["selection"]["k"], **sel_params)
        random.shuffle(parents)

        offspring = []
        for i in range(0, len(parents) - 1, 2):
            c1, c2 = cross(parents[i], parents[i + 1], **cx_params)
            offspring.append(c1)
            offspring.append(c2)

        offspring = [
            mutate(
                ind, width, height,
                **_filter_kwargs(mutate, {
                    **mut_params,
                    "generation": gen,
                    "max_generations": MAX_GENERATIONS,
                }),
            )
            for ind in offspring
        ]

        evaluate_population(offspring, target, width, height)
        population = replace(population, offspring)
        best_per_gen.append(max(population).fitness)

    return best_per_gen


# ── Gráfico de líneas ─────────────────────────────────────────────────────────

def plot_comparison(histories: dict[float, list[float]], mutation_name: str,
                    img_name: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for rate, history in histories.items():
        gens = range(len(history))
        ax.plot(gens, history, color=RATE_COLORS[rate],
                label=f"mutation_rate = {rate}", linewidth=1.8)

    ax.set_xlabel("Generación", fontsize=13)
    ax.set_ylabel("Mejor fitness promedio", fontsize=13)
    ax.set_title(
        f"{MUTATION_LABELS[mutation_name]} — Ruleta + Uniforme — {img_name}",
        fontsize=14, pad=12,
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("output/mutation_rate", exist_ok=True)

    plt.rcParams.update({"font.size": 13, "figure.dpi": 150})

    target_img = Image.open(IMAGE_PATH).convert("RGB")
    width, height = target_img.size
    target = np.array(target_img, dtype=np.float32)

    for mut_method, cfg in CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"  Mutación: {MUTATION_LABELS[mut_method]}")
        print(f"{'=' * 60}")

        histories: dict[float, list[float]] = {}

        for rate in MUTATION_RATES:
            print(f"\n  mutation_rate = {rate}")
            run_histories = []

            for run_idx in range(NUM_RUNS):
                seed = BASE_SEED + run_idx
                history = run_ga(cfg, mut_method, rate, target, width, height, seed=seed)
                run_histories.append(history)
                print(f"    Run {run_idx + 1}/{NUM_RUNS}: fitness final = {history[-1]:.6f}")

            max_len = max(len(h) for h in run_histories)
            padded = [h + [h[-1]] * (max_len - len(h)) for h in run_histories]
            avg_history = list(np.mean(padded, axis=0))
            histories[rate] = avg_history

        save_path = f"output/mutation_rate/mutacion_{mut_method}.png"
        plot_comparison(histories, mut_method, IMAGE_NAME, save_path)

    print(f"\n{'=' * 60}")
    print("  COMPARACIÓN COMPLETA. Resultados en output/mutation_rate/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
