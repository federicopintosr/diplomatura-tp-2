# TP2 - Predicción de fumadores (`smoking`)

Modelo de clasificación que predice si una persona fuma a partir de datos de un chequeo de salud (medidas corporales, presión, análisis de sangre y orina, salud bucal).

**Resultado:** XGBoost con **F1 de la clase 1 = 0,715** en validación (IC 95%: 0,705 - 0,727), medido sobre personas que el modelo nunca vio.

---

## Objetivo y métrica

- Entrenar y validar un modelo con el dataset etiquetado (~50.000 registros).
- Aplicar **el mismo pipeline de preparación** al dataset sin etiquetar y generar la predicción `0` (no fuma) / `1` (fuma).
- Métrica de éxito: **F1-score de la clase 1** (fumadores).

La consigna original está en [`TP2_Modulo_2.ipynb`](TP2_Modulo_2.ipynb).

## Datos

| Archivo | Filas × columnas | Uso |
|---|---|---|
| `data/raw/smoking_prediction.xlsx` | 50.000 × 27 (incluye `smoking`) | Entrenamiento y validación |
| `data/raw/smoking_prediction_entrega.xlsx` | 5.692 × 26 (sin `smoking`) | Predicción final |

- **Unidad muestral:** una persona (un chequeo de salud).
- **Variable objetivo:** `smoking` (1 = fuma). **36,7% de fumadores**: desbalance moderado.
- **Variables:** 24 predictoras: `gender`, `age`, altura, peso, cintura, vista y audición (izq./der.), presión sistólica y diastólica, glucosa, colesterol total, triglicéridos, HDL, LDL, hemoglobina, proteína en orina, creatinina, AST, ALT, GTP, caries y sarro. `ID` es un identificador y `oral` es constante: no entran al modelo.

## Estructura del repositorio

```text
diplomatura-tp-2/
├── data/
│   ├── raw/                  # Excel originales, sin modificar
│   ├── processed/            # train.csv, valid.csv, entrega.csv, columnas.json y predicciones
│   └── external/             # (sin uso)
├── models/
│   ├── modelo_final.joblib   # XGBoost entrenado
│   ├── umbral.json           # configuración, umbral de decisión y features
│   ├── resultados_cv.csv     # todos los experimentos de la notebook 04
│   └── metricas_validacion.json
├── notebooks/
│   ├── 01_lectura_y_discovery.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocesamiento.ipynb
│   ├── 04_entrenamiento_y_optimizacion.ipynb
│   ├── 05_validacion.ipynb
│   ├── 06_prediccion.ipynb
│   └── utils.py              # limpieza compartida por la 03 (train) y la 06 (entrega)
├── TP2_Modulo_2.ipynb        # consigna
├── requirements.txt
└── README.md
```

## Instalación

Requiere **Python 3.10**.

```bash
git clone https://github.com/federicopintosr/diplomatura-tp-2.git
cd diplomatura-tp-2
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` incluye `ipykernel`, suficiente para correr las notebooks desde VS Code. Para usar la interfaz web de Jupyter: `pip install notebook`.

## Ejecución

Las notebooks se ejecutan **desde la carpeta `notebooks/`** (usan rutas relativas) y **en orden**, porque cada una usa lo que guardó la anterior:

| Notebook | Qué hace | Lee | Guarda |
|---|---|---|---|
| 01 | Lectura y diagnóstico de calidad (no modifica nada) | `data/raw/` | — |
| 02 | Análisis exploratorio | `data/raw/` | — |
| 03 | Limpieza, duplicados, split train/validación, experimentos de preprocesamiento | `data/raw/` | `data/processed/` |
| 04 | Comparación de modelos, hiperparámetros, umbral, modelo final | `data/processed/` | `models/` |
| 05 | Validación final (única vez que se usa `valid.csv`) | `data/processed/`, `models/` | `models/metricas_validacion.json` |
| 06 | Predicción sobre el archivo sin etiquetar | `data/raw/`, `models/`, `data/processed/entrega.csv` (control) | `data/processed/` (predicciones) |

Todas las operaciones aleatorias usan `random_state=42`.

**Para reproducir solo las predicciones**, alcanza con ejecutar `06_prediccion.ipynb`: usa el modelo ya entrenado de `models/`.

## Principales hallazgos

**Calidad de los datos**
- **Error de carga del Excel:** 18 columnas numéricas fueron interpretadas como horas `h.mm` y guardadas como fracción de día (por ejemplo, `systolic` 120 → 5,0; `eyesight` 0,8 → 0,0056). Se verificó en las 18 columnas y en los dos archivos (todo valor × 1440 da minutos enteros, siempre entre 0 y 9) y se revierte de forma exacta.
- **Duplicados:** 5.448 pares de filas con las 24 mediciones idénticas y distinto `ID`, siempre con el mismo `smoking`: es la misma persona registrada dos veces. **El 100% de las filas del archivo de entrega es copia de una persona etiquetada.**
- Sin nulos. `oral` es constante. `eyesight = 9.9` es un código de ceguera, no una medición.

**Análisis exploratorio**
- **`gender` es la señal dominante:** fuman el 55% de los hombres y el 4% de las mujeres.
- `height` y `hemoglobin` (correlación ≈ 0,40 con el target) son sobre todo reflejo del sexo: dentro de cada género casi no se relacionan con fumar.
- Señales propias débiles: triglicéridos, GTP, sarro y caries. La edad tiene un efecto no lineal que depende del género.
- Variables de laboratorio (ALT, AST, LDL, GTP, glucosa) con asimetría fuerte y valores extremos clínicamente posibles.
- Train y entrega tienen la misma distribución (sin drift relevante).

## Limpieza y preprocesamiento

Toda la limpieza vive en `utils.preparar`, **una sola función** que aplican la 03 al dataset etiquetado y la 06 al de entrega. Es fila por fila (no calcula estadísticos del conjunto), así que no hay fuga de información.

| Tema | Decisión |
|---|---|
| Error del Excel | Se revierte en las 18 columnas afectadas; la función corta con error si un valor no cumple el patrón |
| `eyesight = 9.9` | Se recodifica a 0,0 (ceguera) |
| `oral` | Se descarta (constante) |
| `gender`, `tartar` | 0/1 con mapeo explícito; corta con error ante un valor no visto |
| Nulos | No hay; si aparecen, la limpieza corta con error en lugar de imputar en silencio |
| Outliers | Se mantienen (casos clínicos reales) |
| Duplicados | Se conservan, pero **todo split y toda validación cruzada se hacen por grupo**: las dos copias de una persona quedan siempre del mismo lado |
| Split | 80/20 estratificado y por grupo (`StratifiedGroupKFold`): 39.999 / 10.001 filas, 0 personas compartidas |
| Desbalance | Sin remuestreo: pesos de clase y umbral de decisión |

## Features y transformaciones

- Se usan las **24 variables originales** en unidades clínicas.
- **Variables nuevas probadas y descartadas:** IMC, cintura/altura, tensión diferencial y los cocientes colesterol/HDL, LDL/HDL, triglicéridos/HDL y AST/ALT. Todas bajaron el F1 en validación cruzada (−0,002 a −0,004).
- **`log1p`** en las variables asimétricas y **`StandardScaler`**: solo para los modelos lineales y KNN, dentro del pipeline (se ajustan con los folds de entrenamiento). Los árboles no los necesitan.

## Modelos y comparación

Validación cruzada **estratificada y por grupo de 5 folds** sobre train, todos los modelos con los mismos folds, optimizando F1 de la clase 1. La brecha train − CV indica sobreajuste.

| Modelo | Configuración | F1 train | F1 CV | Brecha |
|---|---|---|---|---|
| Línea base (todos fuman) | — | 0,537 | 0,537 | 0,000 |
| KNN | k = 201 | 0,671 | 0,665 | 0,007 |
| Árbol de decisión | max_depth = 4 | 0,706 | 0,705 | 0,001 |
| Regresión logística | C = 0,01 | 0,712 | 0,710 | 0,001 |
| Random Forest | 300 árboles, min_samples_leaf = 10 | 0,820 | 0,714 | 0,106 |
| **XGBoost** | búsqueda aleatoria (40 combinaciones) | **0,735** | **0,718** | **0,018** |

Con parámetros por defecto los árboles sobreajustaban (árbol de decisión: 1,000 en train vs 0,564 en CV; XGBoost: 0,831 vs 0,700). Todos los experimentos están en `models/resultados_cv.csv`.

## Modelo elegido

**XGBoost**: árboles entrenados en secuencia, donde cada uno corrige los errores de los anteriores.

- Configuración: 600 árboles de profundidad 4, `learning_rate=0,0195`, `min_child_weight=10`, `subsample=0,67`, `colsample_bytree=0,76`, `gamma=0,83`, `reg_lambda=0,31`.
- Desbalance: `scale_pos_weight=1,73` (no fumadores / fumadores).
- **Umbral de decisión: 0,49**, elegido con predicciones out-of-fold sobre train (sin mirar validación).
- Por qué: mejor F1 en CV con poca brecha. Random Forest queda cerca en CV, pero memoriza mucho más (brecha 0,106).

## Métricas

| Conjunto | F1 clase 1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Validación cruzada (train) | 0,718 | — | — | — |
| Train | 0,732 | 0,615 | 0,905 | 0,870 |
| **Validación (10.001 personas nuevas)** | **0,715** | **0,599** | **0,887** | **0,843** |

- **IC 95% del F1 en validación** (bootstrap, 1.000 remuestreos): 0,705 - 0,727. La estimación de la CV cae dentro.
- PR-AUC 0,71 (azar: 0,37). Accuracy 0,74 (no es la métrica: predecir siempre "no fuma" ya da 63%).
- Como todas las filas de la entrega son copias de personas etiquetadas, el F1 esperado en la entrega es ≈ 0,74.

## Conclusiones

- El modelo detecta **89% de los fumadores** con una precisión del 60%: ante la duda marca "fuma", que es lo que maximiza el F1 de la clase 1.
- **No hay sobreajuste:** train 0,732 vs validación 0,715, y la validación coincide con lo estimado por CV.
- La señal principal es `gender`, seguida de `gtp`. Las probabilidades predichas forman dos grupos: mujeres (casi todas debajo de 0,2) y hombres.
- **Agrupar los duplicados fue clave:** con un split común, ~18% de la validación serían personas que el modelo ya vio, y el F1 quedaría inflado.

## Limitaciones y posibles mejoras

- **Mujeres fumadoras:** el modelo casi no las detecta (3 de 158 en validación). Con 4% de fumadoras, al modelo le conviene clasificarlas a todas como no fumadoras.
- **Falsas alarmas en hombres:** el 75% de los hombres no fumadores queda por encima del umbral. Dentro de los hombres, las variables separan poco.
- Mejoras posibles: un modelo o umbral por género, variables con más información sobre hábitos, calibrar las probabilidades si se usan como riesgo.

## Reproducir las predicciones

1. Instalar dependencias (ver [Instalación](#instalación)).
2. Ejecutar `notebooks/06_prediccion.ipynb` de punta a punta (desde `notebooks/`).
3. Se generan en `data/processed/`:
   - `smoking_prediction_entrega_con_prediccion.xlsx` (y `.csv`): el archivo recibido con la columna **`smoking_prediction`** (0/1) agregada al final.
   - `predicciones.csv`: versión mínima con las columnas `id` y `smoking`.

La 06 no reentrena: carga `models/modelo_final.joblib` y el umbral de `models/umbral.json`, aplica `utils.preparar` y verifica filas, orden, columnas y valores antes de exportar. La salida es determinística.
