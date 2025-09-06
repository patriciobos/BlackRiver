# BlackRiver - Mapas TL

## Requisitos

- Python 3.10 o superior
- Git (opcional)

## 1. Crear y activar entorno virtual

Abre una terminal en la carpeta del proyecto y ejecuta:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Instalar dependencias

Instala todas las dependencias desde `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 3. Procesar mapas

Coloca tus archivos CSV en la carpeta `input-data/` con nombres como:
```
gsm-planta_2023-02-15_f=10 Hz.csv
gsm-transito_2023-02-15_f=100 Hz.csv
```

Ejecuta el script principal para generar los mapas en la carpeta `mapas/`:

```bash
python3 generar_mapas_tl.py
```

El script usa todos los núcleos disponibles y detecta automáticamente el tipo de punto, frecuencia y estación (verano/invierno) según el nombre del archivo.

---

**Nota:** Si tienes problemas con dependencias, asegúrate de tener las librerías del sistema necesarias para Basemap y pyshp.