# Datos

No se suben imágenes al repositorio. Solo se versiona la partición (`splits/pad_split.csv`).

## PAD-UFES-20 (principal)

- Fuente: Pacheco et al., *Data in Brief* 32 (2020), https://doi.org/10.17632/zr7vgbcyr2.1
- Licencia: CC BY 4.0
- 2 298 imágenes clínicas de smartphone, 1 373 pacientes, 6 clases: ACK, BCC, MEL, NEV, SCC, SEK.

```bash
mkdir -p data/raw/pad_download
curl -L -o data/raw/pad_download/pad-ufes-20.zip "https://data.mendeley.com/public-api/zip/zr7vgbcyr2/download/1"   # 3.6 GB
unzip data/raw/pad_download/pad-ufes-20.zip -d data/raw/pad_download
# el zip trae zips anidados de imágenes; dejarlos en data/raw/pad/ junto a metadata.csv
```

Estructura esperada: `data/raw/pad/metadata.csv` y los `*.png` en cualquier subcarpeta de `data/raw/pad/`.

## Partición

`python -m src.data` genera `data/splits/pad_split.csv`: ~70/15/15 train/val/test, estratificado por clase y **agrupado por `patient_id`**. Ningún paciente aparece en dos particiones (la función lo verifica con `assert`). La partición es única y fija (semilla 42); las semillas de los experimentos solo cambian el entrenamiento.

## HAM10000 (dominio dermatoscópico, experimento de brecha)

- Fuente: Tschandl et al., *Scientific Data* 5 (2018), https://doi.org/10.7910/DVN/DBW86T
- Licencia: CC BY-NC 4.0 (uso académico)
- Pendiente de descarga (~2.8 GB, `HAM10000_images_part_{1,2}.zip` + `HAM10000_metadata.tab`).
