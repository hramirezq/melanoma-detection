# Clasificación de lesiones cutáneas robusta a variaciones fotográficas

Investigación: clasificación de lesiones cutáneas y detección de melanoma, robusta a variaciones fotográficas, mediante aprendizaje profundo.

Este proyecto clasifica lesiones cutáneas en fotos clínicas de smartphone (PAD-UFES-20: 6 clases, incluido el melanoma). Mide la robustez de una CNN y de un ViT frente a variaciones de adquisición: iluminación, desenfoque, ruido, compresión y vello.

> Herramienta de investigación. No es un dispositivo médico ni sustituye el diagnóstico de un especialista.

## Instalación

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Datos

Ver [data/README.md](data/README.md) para la descarga. Luego, generar la partición por paciente:

```bash
.venv/bin/python -m src.data --root data/raw/pad --out data/splits/pad_split.csv
```

## Exploración

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
```

Figuras en `results/figures/eda_*.png`.

## Corrupciones

```bash
.venv/bin/python -m src.corruptions
```

Genera `results/figures/corruptions_grid.png` (6 corrupciones × 5 severidades) y comprueba que son deterministas.

## Entrenamiento

Copia redimensionada (lado corto 256 px, PNG) para entrenar rápido:

```bash
.venv/bin/python -m src.data --root data/raw/pad --resize data/raw/pad_256
```

Una corrida (las 18 configs están en `configs/`, `<modelo>_<aug>_s<semilla>.yaml`):

```bash
.venv/bin/python -m src.train configs/effb0_standard_s1.yaml
```

Escribe `results/logs/<run_id>.json`, `results/checkpoints/<run_id>.pt` y `results/preds/<run_id>.csv` (val limpio + test limpio y con 6 corrupciones × 5 severidades). `--epochs 1` sirve de prueba de humo; `--resume` continúa una corrida cortada. En GPU, usar [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) en Colab.

## Evaluación

```bash
.venv/bin/python -m src.baseline            # ResNet-50 congelada + regresión logística
.venv/bin/python -m src.evaluate            # tablas y figura a partir de results/preds/*.csv
.venv/bin/python -m src.evaluate --selftest # predicciones sintéticas: perfecta → AUC 1, aleatoria → ~0,5
```

Genera `results/tables/data.tex` (resumen por partición), `main.tex` (AUC cáncer con IC 95 % por bootstrap de pacientes y especificidad a sensibilidad 90 %, umbral fijado en `val`), `main_6class.tex`, `robustness.tex`, `paired.tex` (diferencias emparejadas de AUC entre augmentations y entre modelos, limpia y con corrupción), `subtype.tex` (sensibilidad por BCC/MEL/SCC), `bias.tex` (por fototipo de Fitzpatrick), `cost.tex` (a partir de `results/logs`), `melanoma_cases.csv`, los CSV correspondientes, `results/figures/robustness_curve.png` y `results/figures/confusion_robust.png`. Requiere `data/raw/pad_256/metadata.csv`.

## Interpretabilidad

```bash
.venv/bin/python -m src.explain   # requiere results/checkpoints/{effb0,vits16}_{standard,robust}_s1.pt
```

Genera `results/figures/gradcam_effb0.png` (Grad-CAM de la probabilidad de cáncer), `results/figures/rollout_vits16.png` (attention rollout), `results/tables/explain_stability.tex` (correlación entre el mapa limpio y con corrupción sobre los cánceres de test) e `inference.tex` (tiempo de inferencia por imagen).

## Demo web

```bash
.venv/bin/python -m app   # requiere results/checkpoints/vits16_robust_s1.pt; abre http://localhost:7860
```

Web local para subir la foto de una lesión y obtener cáncer / no cáncer con ViT-S/16 `robust`, al umbral de sensibilidad 90 % fijado en `val` (`results/tables/per_run.csv`). Muestra las probabilidades de las 6 clases y el attention rollout, y permite degradar la foto con las corrupciones de test para ver la robustez. `--run <run_id>` cambia el modelo. Solo escucha en `127.0.0.1`. No es un dispositivo médico.

## Artículo

```bash
cd paper && tectonic main.tex   # o latexmk -pdf main.tex
```

`paper/main.tex` (IEEEtran) incluye las tablas de `results/tables/*.tex` y las figuras de `results/figures/` directamente, así que se compila después de regenerarlas.

## Estructura

```
data/       README de descarga + partición versionada
src/        datos, modelos, entrenamiento y evaluación
configs/    hiperparámetros por experimento
notebooks/  exploración de datos
results/    tablas, figuras y logs del artículo
```
