"""
analysis.py — Generación automática de gráficos y métricas para presentación.

Ejecutar:  python analysis.py

Se generan imágenes en  output/analysis/  organizadas por categoría:
    selection/    replacement/    crossover/    mutation/

Cada categoría contiene:
    1. Evolución temporal del fitness por método (1 corrida representativa).
    2. Gráfico de barras con diferencia de fitness (final − inicial) con barras de error
       (promedio ± desviación estándar sobre múltiples corridas).
    3. Imágenes renderizadas del mejor individuo de la corrida representativa.

Todos los textos en español.  Sin títulos.  Fuentes grandes para diapositivas.
"""

import argparse
import copy
import json
import os
import random
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")  # backend sin GUI
import matplotlib.pyplot as plt
from PIL import Image

# ─── Configuración de rutas ──────────────────────────────────────────────────
sys.path.insert(0, "src")

import crossover as cx
import mutation as mut
import replacement as rep
import selection as sel
from fitness import evaluate_population, render
from individual import Individual
from population import initialize


def load_config(path: str = "config-analysis.json") -> dict:
    with open(path) as f:
        return json.load(f)


# ─── Mapas de métodos ────────────────────────────────────────────────────────
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

# ─── Estilo global de matplotlib ─────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

# Paleta de colores distinguibles
COLORS = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Función de ejecución del AG (una corrida)
# ═══════════════════════════════════════════════════════════════════════════════


def run_ga(
    target: np.ndarray,
    width: int,
    height: int,
    select_fn,
    cross_fn,
    mutate_fn,
    replace_fn,
    config: dict,
    seed: int | None = None,
) -> tuple[list[float], Individual]:
    """Ejecuta el AG y devuelve (historial_fitness, mejor_individuo)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    pop_size = config["population_size"]
    num_tri = config["num_triangles"]
    max_gen = config["max_generations"]
    sel_k = config["selection"]["k"]
    mut_params = config["mutation"]

    stagnation_n = config.get("stagnation_generations")
    stagnation_eps = config.get("stagnation_epsilon", 1e-6)
    use_stagnation = stagnation_n is not None and stagnation_n > 0

    population = initialize(pop_size, num_tri, width, height)
    evaluate_population(population, target, width, height)

    best = max(population)
    history = [best.fitness]
    best_ever = best.fitness
    gens_without_gain = 0

    for gen in range(1, max_gen + 1):
        parents = select_fn(population, sel_k)
        random.shuffle(parents)

        offspring = []
        for i in range(0, len(parents) - 1, 2):
            c1, c2 = cross_fn(parents[i], parents[i + 1])
            offspring.append(c1)
            offspring.append(c2)

        offspring = [
            mutate_fn(
                ind,
                width,
                height,
                generation=gen,
                max_generations=max_gen,
                **mut_params,
            )
            for ind in offspring
        ]

        evaluate_population(offspring, target, width, height)
        population = replace_fn(population, offspring)

        best = max(population)
        history.append(best.fitness)

        if use_stagnation:
            if best.fitness > best_ever + stagnation_eps:
                best_ever = best.fitness
                gens_without_gain = 0
            else:
                gens_without_gain += 1
                if best.fitness > best_ever:
                    best_ever = best.fitness
                if gens_without_gain >= stagnation_n:
                    break

    best_overall = max(population)
    return history, best_overall


# ═══════════════════════════════════════════════════════════════════════════════
# Funciones de graficado
# ═══════════════════════════════════════════════════════════════════════════════


def plot_temporal_evolution(
    histories: dict[str, list[float]],
    save_path: str,
):
    """Gráfico de líneas: evolución temporal del fitness por método."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, (name, hist) in enumerate(histories.items()):
        color = COLORS[idx % len(COLORS)]
        ax.plot(range(len(hist)), hist, label=name, color=color, linewidth=2)

    ax.set_xlabel("Generación")
    ax.set_ylabel("Mejor fitness")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ Guardado: {save_path}")


def plot_fitness_delta_bars(
    deltas: dict[str, list[float]],
    save_path: str,
):
    """Gráfico de barras con barras de error: Δfitness = final − inicial."""
    names = list(deltas.keys())
    means = [np.mean(v) for v in deltas.values()]
    stds = [np.std(v) for v in deltas.values()]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 7))

    x = np.arange(len(names))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        color=[COLORS[i % len(COLORS)] for i in range(len(names))],
        edgecolor="black",
        linewidth=0.7,
        alpha=0.85,
        error_kw={"elinewidth": 2, "capthick": 2},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Δ fitness (final − inicial)")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ Guardado: {save_path}")


def save_result_image(
    individual,
    width: int,
    height: int,
    save_path: str,
):
    """Renderiza el mejor individuo y guarda la imagen PNG."""
    img = render(individual, width, height).convert("RGB")
    img.save(save_path)
    print(f"  ✓ Guardado: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Bloque principal de análisis
# ═══════════════════════════════════════════════════════════════════════════════


def analyse_category(
    category: str,
    methods: dict,
    target: np.ndarray,
    width: int,
    height: int,
    build_config,
):
    """
    Ejecuta el análisis completo para una categoría de operadores.

    build_config(name, method) → (select_fn, cross_fn, mutate_fn, replace_fn, cfg)
    """
    out_dir = os.path.join("output", "analysis", category)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Corrida representativa por método → evolución temporal y resultado
    print(f"\n{'═' * 60}")
    print(f"  Categoría: {category.upper()}")
    print(f"{'═' * 60}")

    histories: dict[str, list[float]] = {}
    best_individuals: dict[str, object] = {}

    for name, method in methods.items():
        print(f"  ▸ Corrida representativa: {name} …")
        select_fn, cross_fn, mutate_fn, replace_fn, cfg = build_config(name, method)
        hist, best_ind = run_ga(
            target,
            width,
            height,
            select_fn,
            cross_fn,
            mutate_fn,
            replace_fn,
            cfg,
            seed=BASE_SEED,
        )
        histories[name] = hist
        best_individuals[name] = best_ind
        print(f"    fitness final = {hist[-1]:.6f}")

    # Gráfico de evolución temporal
    plot_temporal_evolution(
        histories,
        os.path.join(out_dir, f"{category}_evolucion_temporal.png"),
    )

    # Imágenes del mejor resultado por método
    for name, ind in best_individuals.items():
        safe_name = name.replace(" ", "_").replace(".", "").lower()
        save_result_image(
            ind,
            width,
            height,
            os.path.join(out_dir, f"{category}_resultado_{safe_name}.png"),
        )

    # 2. Múltiples corridas para barras de error
    print(f"\n  ▸ {NUM_RUNS_ERROR_BARS} corridas por método para barras de error …")
    deltas: dict[str, list[float]] = {name: [] for name in methods}

    for run_idx in range(NUM_RUNS_ERROR_BARS):
        seed = BASE_SEED + run_idx + 1
        for name, method in methods.items():
            select_fn, cross_fn, mutate_fn, replace_fn, cfg = build_config(name, method)
            hist, _ = run_ga(
                target,
                width,
                height,
                select_fn,
                cross_fn,
                mutate_fn,
                replace_fn,
                cfg,
                seed=seed,
            )
            delta = hist[-1] - hist[0]
            deltas[name].append(delta)
        print(f"    corrida {run_idx + 1}/{NUM_RUNS_ERROR_BARS} completada")

    plot_fitness_delta_bars(
        deltas,
        os.path.join(out_dir, f"{category}_delta_fitness.png"),
    )

    print(f"  ✓ Categoría {category} completa.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Constructores de configuración por categoría
# ═══════════════════════════════════════════════════════════════════════════════


def _load_target():
    target_img = Image.open(IMAGE_PATH).convert("RGB")
    width, height = target_img.size
    target = np.array(target_img, dtype=np.float32)
    return target, width, height


def _base_cfg():
    return copy.deepcopy(BASE_CONFIG)


# ── Selección ────────────────────────────────────────────────────────────────


def build_selection(name, method):
    """
    Varía el método de selección.
    Fijo: cruza=un punto · mutación=uniforme (tasa=0.01) · reemplazo=tradicional
    """
    cfg = _base_cfg()
    select_fn = method
    cross_fn = cx.one_point
    mutate_fn = mut.uniform
    replace_fn = rep.traditional
    return select_fn, cross_fn, mutate_fn, replace_fn, cfg


# ── Reemplazo ────────────────────────────────────────────────────────────────


def build_replacement(name, method):
    """
    Varía el método de reemplazo.
    Fijo: selección=élite · cruza=un punto · mutación=uniforme (tasa=0.01)
    """
    cfg = _base_cfg()
    select_fn = sel.elite
    cross_fn = cx.one_point
    mutate_fn = mut.uniform
    replace_fn = method
    return select_fn, cross_fn, mutate_fn, replace_fn, cfg


# ── Cruza ────────────────────────────────────────────────────────────────────


def build_crossover(name, method):
    """
    Varía el método de cruza.
    Fijo: selección=élite · mutación=uniforme (tasa=0.01) · reemplazo=tradicional
    """
    cfg = _base_cfg()
    select_fn = sel.elite
    cross_fn = method
    mutate_fn = mut.uniform
    replace_fn = rep.traditional
    return select_fn, cross_fn, mutate_fn, replace_fn, cfg


# ── Mutación ─────────────────────────────────────────────────────────────────


def build_mutation(name, method):
    """
    Varía el método de mutación.
    Fijo: selección=élite · cruza=un punto · reemplazo=tradicional
    """
    cfg = _base_cfg()
    select_fn = sel.elite
    cross_fn = cx.one_point
    mutate_fn = method
    replace_fn = rep.traditional
    return select_fn, cross_fn, mutate_fn, replace_fn, cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    print("Cargando imagen objetivo …")
    target, width, height = _load_target()
    print(f"  Imagen: {IMAGE_PATH}  ({width}×{height})")
    print(
        f"  Población: {BASE_CONFIG['population_size']}  |  "
        f"Triángulos: {BASE_CONFIG['num_triangles']}  |  "
        f"Generaciones: {BASE_CONFIG['max_generations']}"
    )
    print(f"  Corridas para barras de error: {NUM_RUNS_ERROR_BARS}")

    # ── 1. Selección ─────────────────────────────────────────────────────
    analyse_category(
        "selection",
        SELECTION_METHODS,
        target,
        width,
        height,
        build_selection,
    )

    # ── 2. Reemplazo ─────────────────────────────────────────────────────
    analyse_category(
        "replacement",
        REPLACEMENT_METHODS,
        target,
        width,
        height,
        build_replacement,
    )

    # ── 3. Cruza ─────────────────────────────────────────────────────────
    analyse_category(
        "crossover",
        CROSSOVER_METHODS,
        target,
        width,
        height,
        build_crossover,
    )

    # ── 4. Mutación ──────────────────────────────────────────────────────
    analyse_category(
        "mutation",
        MUTATION_METHODS,
        target,
        width,
        height,
        build_mutation,
    )

    print("\n" + "═" * 60)
    print("  ANÁLISIS COMPLETO.  Resultados en output/analysis/")
    print("═" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Análisis comparativo de operadores del AG"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config-analysis.json",
        help="Ruta al archivo de configuración JSON (por defecto: config-analysis.json)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    IMAGE_PATH = cfg["image_path"]
    NUM_RUNS_ERROR_BARS = cfg.get("num_runs_error_bars", 5)
    BASE_SEED = cfg.get("base_seed", 42)

    # Parámetros fijos del AG (salvo la variable bajo análisis).
    # Los operadores se usan como valores fijos ("control") cuando se varía
    # una categoría distinta.  Ej.: al comparar métodos de selección,
    # cruza/mutación/reemplazo se mantienen en estos valores.
    BASE_CONFIG = {
        "num_triangles": cfg["num_triangles"],
        "population_size": cfg["population_size"],
        "max_generations": cfg["max_generations"],
        "stagnation_generations": cfg.get("stagnation_generations"),
        "stagnation_epsilon": cfg.get("stagnation_epsilon", 1e-6),
        "selection": cfg["selection"],
        "crossover": cfg["crossover"],
        "mutation": cfg["mutation"],
        "replacement": cfg["replacement"],
    }

    main()
