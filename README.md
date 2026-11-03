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

## Estructura

```
data/       README de descarga + partición versionada
src/        datos, modelos, entrenamiento y evaluación
configs/    hiperparámetros por experimento
notebooks/  exploración de datos
results/    tablas, figuras y logs del artículo
```
