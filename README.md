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

## 📂 Estructura del Proyecto

```text
.
├── main.py              # Punto de entrada de la aplicación
├── config.json          # Archivo de configuración de parámetros
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
