# Tareas Pendientes (TODO) - SIA TP2

Basado en la revisión del documento de requerimientos (`docs/SIA - TP2 - 2026 1Q.pdf`) y del código fuente (directorio `src` y `main.py`), a continuación se detalla lo que falta implementar o finalizar para completar el Trabajo Práctico 2:

## 1. 🧬 Mutación (`src/mutation.py`)

Aunque se definió la estructura para los diferentes tipos de mutación, varios de ellos levantan `NotImplementedError` y deben ser codificados:

- [X] **Mutación de un Gen (`gene`)**: Debe alterar únicamente un único gen (por ejemplo, el valor `x` de un vértice, el color `R`, etc.) de un triángulo elegido de forma aleatoria, dada la probabilidad de mutación.
- [X] **Mutación Multigen (`multigen`)**: Debe mutar una cantidad determinada de genes (no necesariamente del mismo triángulo).
- [X] **Mutación No Uniforme (`non_uniform`)**: La magnitud o probabilidad de la mutación debe decaer (o cambiar) en función de la generación actual respecto a la cantidad máxima de generaciones (`generation` / `max_generations`).

## 2. 🛑 Criterios de Parada (`main.py`)

Se han implementado la parada por máxima cantidad de generaciones y por contenido (alcanzar un `target_fitness`), pero la consigna pide evaluar métodos adicionales:

- [X] **Parada por Estructura**: Implementar lógica para cortar la ejecución del motor si la población no cambia o si el `best_fitness` no mejora significativamente durante una cantidad $N$ consecutiva de generaciones.

## 3. 📊 Métricas y Análisis (`main.py`)

La consigna exige _métricas para análisis para defender su implementación_. Actualmente el motor imprime por consola el mejor fitness de cada generación pero no guarda los datos.

- [ ] **Almacenamiento de Métricas**: En cada iteración, guardar data (ej. mejor fitness, fitness promedio, error, diversidad) en un arreglo o archivo CSV/JSON.
- [ ] **Gráficos**: Plasmar esos datos en gráficos para adjuntar en la presentación (Ej: Fitness a lo largo de las generaciones).

## 4. 📝 Preguntas de Análisis (Requisito de consigna)

Debe redactarse la respuesta a las preguntas del final del PDF ANTES de la etapa de experimentación (Idealmente en un `INFORME.md` o en la presentación final):

- [ ] ¿Cómo evalúo mi aproximación al dibujo?
- [ ] ¿Qué es un individuo en este problema? ¿Cuáles serían sus genes?
- [ ] ¿Qué es el fitness en este problema?
- [ ] ¿Cómo podría mutar un individuo?
- [ ] ¿Cómo podría cruzar individuos para obtener descendencia? ¿Por qué generaría descendientes con buenas probabilidades de mejorar?
- [ ] ¿Cómo sería la versión más simple de esto?
- [ ] ¿Qué tipo de imagen, y sobre todo cómo afecta la cantidad de triángulos a la performance si quiero evaluar rápidamente mi motor de AG?
- [ ] ¿Alcanza implementar PARCIALMENTE los requerimientos para evaluar el motor?

## 5. 🎯 Entregables Finales

- [ ] Redactar la justificación de la estructura elegida y de la función de aptitud.
- [ ] Decidir y justificar qué método(s) de cruza y mutación usarían bajo diferentes circunstancias.
- [ ] Preparar la **Presentación** en formato digital (Diapositivas).
- [ ] Validar que todo el código fuente cumple los requerimientos.
