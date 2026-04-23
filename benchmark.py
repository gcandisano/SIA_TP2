"""
benchmark.py — Comparación de 5 configuraciones del AG con gráfico de barras.

Genera un gráfico por imagen (argentina, colombia, chrome).
Cada barra = promedio del mejor fitness al final de 500 generaciones sobre 3 corridas.

Ejecutar:  python benchmark.py
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
from individual import Individual
from population import initialize


# ── Parámetros globales ───────────────────────────────────────────────────────

NUM_RUNS = 3
BASE_SEED = 42
MAX_GENERATIONS = 500
POPULATION_SIZE = 50
NUM_TRIANGLES = 50

IMAGES = {
    "Argentina": "assets/argentina_flag.png",
    "Colombia": "assets/colombia.png",
    "Chrome": "assets/chrome.png",
}

# ── 5 Configuraciones representativas ────────────────────────────────────────
#
#  Cada config varía selección + cruza + mutación + reemplazo para cubrir
#  distintas estrategias del espectro:
#
#  1. Élite + Un punto + Uniforme           → baseline conservador
#  2. Torneo det. + Cruce uniforme + No-uniforme → presión alta, mutación adaptiva
#  3. Ruleta + Anular + Por gen             → selección proporcional, mutación puntual
#  4. Ranking + Dos puntos + Multigen       → ranking + reemplazo joven
#  5. Boltzmann + Cruce uniforme + Completa → exploración temprana, explotación tardía

CONFIGS = {
    "Élite\nUn punto\nUniforme": {
        "selection": {"method": "elite", "k": POPULATION_SIZE},
        "crossover": {"method": "one_point"},
        "mutation":  {"method": "uniform",     "mutation_rate": 0.01},
        "replacement": {"method": "traditional"},
    },
    "Torneo\nCruce unif.\nNo-uniforme": {
        "selection": {"method": "tournament_det", "k": POPULATION_SIZE, "tournament_size": 5},
        "crossover": {"method": "uniform",  "swap_prob": 0.5},
        "mutation":  {"method": "non_uniform", "mutation_rate": 0.1, "b": 5.0},
        "replacement": {"method": "traditional"},
    },
    "Ruleta\nAnular\nPor gen": {
        "selection": {"method": "roulette", "k": POPULATION_SIZE},
        "crossover": {"method": "annular"},
        "mutation":  {"method": "gene", "mutation_rate": 0.1},
        "replacement": {"method": "traditional"},
    },
    "Ranking\nDos puntos\nMultigen": {
        "selection": {"method": "ranking", "k": POPULATION_SIZE},
        "crossover": {"method": "two_point"},
        "mutation":  {"method": "multigen", "num_genes": 3},
        "replacement": {"method": "young_bias"},
    },
    "Boltzmann\nCruce unif.\nCompleta": {
        "selection": {"method": "boltzmann", "k": POPULATION_SIZE, "temperature": 1.0},
        "crossover": {"method": "uniform", "swap_prob": 0.5},
        "mutation":  {"method": "complete", "mutation_rate": 0.05},
        "replacement": {"method": "young_bias"},
    },
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

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filter_kwargs(func, kwargs: dict) -> dict:
    sig = inspect.signature(func)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def run_ga(cfg: dict, target: np.ndarray, width: int, height: int, seed: int | None = None) -> float:
    """Ejecuta el AG y devuelve el mejor fitness al final."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    select  = SELECTION_METHODS[cfg["selection"]["method"]]
    cross   = CROSSOVER_METHODS[cfg["crossover"]["method"]]
    mutate  = MUTATION_METHODS[cfg["mutation"]["method"]]
    replace = REPLACEMENT_METHODS[cfg["replacement"]["method"]]

    sel_params = _filter_kwargs(select, {k: v for k, v in cfg["selection"].items() if k not in ("method", "k")})
    cx_params  = _filter_kwargs(cross,  {k: v for k, v in cfg["crossover"].items() if k != "method"})
    mut_params = {k: v for k, v in cfg["mutation"].items() if k != "method"}

    sel_k = cfg["selection"]["k"]

    population = initialize(POPULATION_SIZE, NUM_TRIANGLES, width, height)
    evaluate_population(population, target, width, height)

    for gen in range(1, MAX_GENERATIONS + 1):
        parents = select(population, sel_k, **sel_params)
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

    return max(population).fitness


# ── Gráfico de barras ─────────────────────────────────────────────────────────

def plot_bar_chart(config_names: list[str], avg_fitnesses: list[float], img_name: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(config_names))
    bars = ax.bar(
        x,
        avg_fitnesses,
        color=COLORS[: len(config_names)],
        edgecolor="black",
        linewidth=0.7,
        alpha=0.88,
        width=0.55,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(config_names, fontsize=11)
    ax.set_ylabel("Mejor fitness promedio (3 corridas)", fontsize=13)
    ax.set_title(f"Comparación de configuraciones — {img_name}", fontsize=14, pad=12)

    y_min = max(0.0, min(avg_fitnesses) * 0.97)
    y_max = max(avg_fitnesses) * 1.04
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, avg_fitnesses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (y_max - y_min) * 0.005,
            f"{val:.5f}",
            ha="center", va="bottom", fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("output/comparison", exist_ok=True)

    plt.rcParams.update({
        "font.size": 13,
        "figure.dpi": 150,
    })

    config_names  = list(CONFIGS.keys())
    config_values = list(CONFIGS.values())

    for img_name, img_path in IMAGES.items():
        print(f"\n{'=' * 60}")
        print(f"  Imagen: {img_name}  ({img_path})")
        print(f"{'=' * 60}")

        target_img = Image.open(img_path).convert("RGB")
        width, height = target_img.size
        target = np.array(target_img, dtype=np.float32)

        avg_fitnesses = []

        for cfg_name, cfg in zip(config_names, config_values):
            label = cfg_name.replace("\n", " / ")
            print(f"\n  Config: {label}")
            fitnesses = []
            for run_idx in range(NUM_RUNS):
                seed = BASE_SEED + run_idx
                fitness = run_ga(cfg, target, width, height, seed=seed)
                fitnesses.append(fitness)
                print(f"    Run {run_idx + 1}/{NUM_RUNS}: fitness = {fitness:.6f}")
            avg = float(np.mean(fitnesses))
            avg_fitnesses.append(avg)
            print(f"    Promedio: {avg:.6f}")

        safe_name = img_name.lower()
        save_path = f"output/comparison/barras_{safe_name}.png"
        plot_bar_chart(config_names, avg_fitnesses, img_name, save_path)

    print(f"\n{'=' * 60}")
    print("  COMPARACIÓN COMPLETA. Resultados en output/comparison/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
