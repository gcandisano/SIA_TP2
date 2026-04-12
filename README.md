# Compresor de Imágenes mediante Algoritmos Genéticos (SIA - TP2)

Este proyecto implementa un motor de **Algoritmos Genéticos (AG)** diseñado para aproximar una imagen objetivo utilizando una cantidad determinada de triángulos sobre un canvas blanco. Es el segundo ejercicio del Trabajo Práctico 2 de la materia **Sistemas de Inteligencia Artificial (SIA)** del ITBA.

## 🚀 Descripción del Problema

El objetivo es encontrar la mejor combinación de triángulos (posición de vértices, color y transparencia) que, al superponerse, minimicen la diferencia visual con una imagen de entrada.

- **Individuo**: Una colección de $N$ triángulos.
- **Genes**: Las propiedades de cada triángulo (3 vértices $(x,y)$, color $RGB$ y transparencia $\alpha$).
- **Fitness**: Métrica basada en el Error Cuadrático Medio ($1 - MSE$) entre la imagen generada y la original.

## 🛠️ Requisitos

- **Python**: 3.12 o superior.
- **Bibliotecas**:
  - `numpy`: Para cálculos matriciales.
  - `pillow` (PIL): Para el renderizado y procesamiento de imágenes.
  - `matplotlib`: Para generar el gráfico de evolución del fitness.

## ⚙️ Instalación

1. **Clonar el repositorio**:

   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd SIA_TP2
   ```

2. **Configurar el entorno**:

   ### Opción A: Usando [uv](https://docs.astral.sh/uv/) (Recomendado)

   ```bash
   uv sync
   ```

   ### Opción B: Usando venv + pip

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## 🖥️ Uso

Para ejecutar el motor, utiliza el archivo `main.py`. Puedes pasar la ruta de un archivo de configuración JSON como argumento (por defecto usa `config.json`).

### Ejecución con uv

```bash
uv run main.py --config ruta/a/config.json
```

### Ejecución con Python (venv activo)

```bash
python main.py --config ruta/a/config.json
```

Para ver la ayuda y opciones disponibles:

```bash
uv run main.py --help
```

o si no usas uv:

```bash
python main.py --help
```

Los resultados se guardarán en la carpeta `output/`:

- `result.png`: mejor aproximación final.
- `best_fitness.png`: evolución del mejor fitness por generación.

## 📄 Configuración (`config.json`)

El archivo de configuración permite ajustar todos los parámetros del algoritmo:

```json
{
  "image_path": "assets/target.png",
  "num_triangles": 100,
  "population_size": 100,
  "max_generations": 2000,
  "target_fitness": 0.95,
  "stagnation_generations": 150,
  "stagnation_epsilon": 1e-6,
  "selection": { "method": "elite", "k": 50 },
  "crossover": { "method": "one_point" },
  "mutation": { "method": "uniform", "mutation_rate": 0.05 },
  "replacement": { "method": "traditional" }
}
```

## 🧬 Métodos Implementados

### Selección

- **Elite**: Selecciona los $k$ individuos con mejor fitness.
- **Ruleta (Roulette)**: Selección proporcional al fitness.
- **Universal**: Selección estocástica universal.
- **Boltzmann**: Basado en temperatura (configurar `"temperature"` en `selection`).
- **Torneos**:
  - `tournament_det`: Determinístico (configurar `"tournament_size"`).
  - `tournament_prob`: Probabilístico (configurar `"threshold"`).
- **Ranking**: Selección basada en la posición relativa en la población.

### Cruce (Crossover)

- **Punto único (`one_point`)**
- **Dos puntos (`two_point`)**
- **Uniforme (`uniform`)** (configurar `"swap_prob"`)
- **Anular (`annular`)**

### Mutación

- **Gen (`gene`)**: Con probabilidad `mutation_rate`, muta un único alelo en un triángulo elegido al azar.
- **MultiGen (`multigen`)**: Muta `num_genes` alelos (con posible reemplazo de triángulos).
- **Uniforme (`uniform`)**: Cada triángulo se reemplaza por uno aleatorio con probabilidad `mutation_rate`.
- **No uniforme (`non_uniform`)**: Similar a `gene`, pero la magnitud del cambio decae con la generación.
- **Completa (`complete`)**: Con probabilidad `mutation_rate`, muta todos los triángulos del individuo.

### Reemplazo

- **Tradicional (`traditional`)**: Supervivencia aditiva. Selecciona los mejores de la unión padres + hijos.
- **Young Bias (`young_bias`)**: Prioriza descendientes. Si `K > N`, selecciona `N` hijos; si `K <= N`, toma `K` hijos y completa con `N-K` padres.

### Criterios de parada

- **Máxima cantidad de generaciones** (`max_generations`).
- **Fitness objetivo** (`target_fitness`).
- **Estancamiento** (`stagnation_generations` y `stagnation_epsilon`).

## 📊 Análisis Comparativo (`analysis.py`)

El script `analysis.py` ejecuta un análisis comparativo completo de todos los operadores del AG y genera gráficos listos para presentación. Lee su configuración desde `config-analysis.json`.

### Ejecución

```bash
# Con uv
uv run analysis.py

# Con uv y archivo de configuración personalizado
uv run analysis.py --config ruta/a/config-analysis.json

# Con Python (venv activo)
python analysis.py

# Con archivo de configuración personalizado
python analysis.py --config ruta/a/config-analysis.json
```

### Configuración (`config-analysis.json`)

Los parámetros se definen en el archivo `config-analysis.json`:

| Parámetro                               | Valor por defecto             | Descripción                               |
| --------------------------------------- | ----------------------------- | ----------------------------------------- |
| `IMAGE_PATH`                            | `"assets/argentina_flag.png"` | Imagen objetivo para la aproximación      |
| `BASE_CONFIG["num_triangles"]`          | `50`                          | Cantidad de triángulos por individuo      |
| `BASE_CONFIG["population_size"]`        | `50`                          | Tamaño de la población                    |
| `BASE_CONFIG["max_generations"]`        | `300`                         | Máximo de generaciones por corrida        |
| `BASE_CONFIG["stagnation_generations"]` | `150`                         | Generaciones sin mejora para detener      |
| `BASE_CONFIG["stagnation_epsilon"]`     | `1e-6`                        | Umbral mínimo de mejora                   |
| `NUM_RUNS_ERROR_BARS`                   | `5`                           | Cantidad de corridas para barras de error |
| `BASE_SEED`                             | `42`                          | Semilla base para reproducibilidad        |

### Parámetros fijos por análisis

Cada análisis varía **un solo tipo de operador** y mantiene el resto fijo:

| Análisis      | Parámetros fijos                                                                |
| ------------- | ------------------------------------------------------------------------------- |
| **Selección** | Cruza: `one_point` · Mutación: `uniform` (tasa=0.01) · Reemplazo: `traditional` |
| **Reemplazo** | Selección: `elite` · Cruza: `one_point` · Mutación: `uniform` (tasa=0.01)       |
| **Cruza**     | Selección: `elite` · Mutación: `uniform` (tasa=0.01) · Reemplazo: `traditional` |
| **Mutación**  | Selección: `elite` · Cruza: `one_point` · Reemplazo: `traditional`              |

### Salida

Los resultados se guardan en `output/analysis/` con la siguiente estructura:

```text
output/analysis/
├── selection/
│   ├── selection_evolucion_temporal.png   # Curvas de fitness por método
│   ├── selection_delta_fitness.png        # Barras Δfitness con error
│   └── selection_resultado_*.png          # Imagen resultado por método
├── replacement/
│   └── ...
├── crossover/
│   └── ...
└── mutation/
    └── ...
```

Cada categoría genera:

1. **Evolución temporal**: gráfico de líneas con el fitness a lo largo de las generaciones (1 corrida representativa).
2. **Δ fitness**: gráfico de barras con la mejora de fitness (final − inicial) promediada sobre múltiples corridas, con barras de error (desviación estándar).
3. **Imágenes resultado**: renderizado del mejor individuo obtenido con cada método.

## 📂 Estructura del Proyecto

```text
.
├── main.py              # Punto de entrada de la aplicación
├── analysis.py          # Script de análisis comparativo
├── config.json          # Configuración del motor principal
├── config-analysis.json # Configuración del análisis comparativo
├── pyproject.toml       # Definición de dependencias
├── assets/              # Carpeta de imágenes de entrada
├── output/              # Resultados generados
├── docs/                # Consigna del trabajo práctico
└── src/                 # Lógica core del algoritmo
    ├── crossover.py     # Métodos de cruce
    ├── fitness.py       # Evaluación y renderizado
    ├── individual.py    # Clase Individual y Triangle
    ├── mutation.py      # Métodos de mutación
    ├── population.py    # Gestión de la población inicial
    ├── replacement.py   # Métodos de supervivencia
    └── selection.py     # Métodos de selección de padres
```

---

**Instituto Tecnológico de Buenos Aires (ITBA)**  
Sistemas de Inteligencia Artificial - 1Q 2026
