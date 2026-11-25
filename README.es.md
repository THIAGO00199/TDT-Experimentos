# Teoría de la Deuda Temporal (TDT)

Investigación computacional independiente sobre irreversibilidad, memoria, información y control en sistemas físicos finitos.

**Estado de la investigación:** marco teórico exploratorio acompañado de simulaciones clásicas y cuánticas. La TDT no se presenta como una ley física establecida ni como una teoría validada universalmente.

## Pregunta central

Este trabajo investiga si las descripciones operativas de la irreversibilidad y la memoria pueden ayudar a caracterizar el costo de transformar, controlar o recuperar estados físicos cuando la información relevante permanece distribuida en el entorno.

La idea se ha examinado en distintos modelos: sistemas de dos niveles, cadenas de Ising, baños finitos, modelos cuánticos con modos explícitos y procesos markovianos y no markovianos. Cada modelo prueba solo una parte de la hipótesis; ningún resultado aislado demuestra la teoría completa.

## Líneas de investigación

- **Termodinámica de la información:** límites de Landauer, borrado e información lateral.
- **Baños finitos y recuperación:** modelos de Ising y del demonio de Creutz mediante simulaciones estocásticas.
- **Memoria cuántica:** modos armónicos explícitos, dinámica reducida y testigos de backflow.
- **Control óptimo:** protocolos numéricos, refinamiento temporal y comparación con referencias convencionales.

## Resultados seleccionados

- Un benchmark de dos niveles resuelve numéricamente la dinámica de la ecuación maestra y verifica el balance de energía con residuos cercanos a la precisión numérica. Al mismo tiempo, la magnitud operacional denominada Pi cambia con la escala de referencia y, en general, no coincide con la disipación recuperable. Esto delimita una cuestión matemática importante.
- Un modelo con dos qubits y dos modos armónicos truncados tiene un espacio de Hilbert de dimensión 36 y muestra backflow de distancia de traza para el par de estados y los parámetros elegidos. El resultado caracteriza la memoria en ese modelo concreto; no es una firma exclusiva de la TDT.
- Una línea de simulaciones estocásticas de baños finitos ejecuta hasta 100.000 trayectorias por condición para comparar estrategias de control, evolución y recuperación en modelos de Ising y del demonio de Creutz.
- Un cálculo exacto de un anillo de Ising de cuatro bits estudia el límite ideal de borrado cuando la magnetización final está disponible como información lateral.

Consulta [docs/EXPERIMENT_CATALOG.md](docs/EXPERIMENT_CATALOG.md) para conocer los métodos, archivos de resultados y limitaciones.

## Contenido del laboratorio

- `*.py` — simulaciones y análisis en Python.
- `*.cpp` — simuladores compilados para experimentos estocásticos.
- `*.json` — resultados estructurados de ejecuciones específicas.
- `*.txt` y `*.log` — informes y registros de ejecución.
- `*.npz`, `*.csv` y `*.png` — artefactos numéricos y visuales de experimentos seleccionados.

Los nombres `v2`, `v4` y `v5` identifican etapas históricas de investigación, no versiones de una teoría consolidada.

## Entorno

- Python 3.12 o posterior.
- NumPy y SciPy para las simulaciones numéricas.
- Matplotlib para los scripts de visualización.
- Un compilador C++ compatible para los experimentos `*.cpp`.

Crea un entorno virtual e instala las dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Reproducibilidad:** consulta primero el catálogo de experimentos. Algunos scripts históricos usan rutas locales absolutas o escriben resultados en el directorio de trabajo. Ejecútalos en una copia limpia del laboratorio para conservar los resultados existentes.

## Integridad científica

- Los resultados negativos, inconclusos y las limitaciones forman parte del registro.
- El backflow, por sí solo, no implica una nueva ley termodinámica.
- La optimización SPSA no demuestra que se haya encontrado un mínimo global.
- Los modelos markovianos y los modelos con modos explícitos solo pueden compararse cuando sus hipótesis y parámetros físicos son compatibles.
- Una identidad numérica o un balance de energía correcto no demuestra que la TDT aporte una predicción nueva.
- La formulación matemática central todavía requiere definiciones operativas consistentes, análisis dimensional y pruebas independientes.

## Estado actual

La TDT sigue siendo un programa de investigación exploratorio. El valor del repositorio está en el conjunto de modelos, códigos, resultados rastreables y preguntas que superaron —o no— las pruebas realizadas. No se afirma validación experimental, consenso científico ni aplicabilidad universal.

---

Leia também: [Português](README.md) · [English](README.en.md) · [Catálogo de experimentos](docs/EXPERIMENT_CATALOG.md)