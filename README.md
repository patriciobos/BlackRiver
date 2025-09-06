
# BlackRiver - Instrucciones de entorno y dependencias

## Instalación de dependencias

### Opción 1: Manual (Linux/Mac/Windows)

1. Abre una terminal en la carpeta del proyecto.
2. Crea y activa el entorno virtual:
	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```
3. Instala las dependencias del proyecto:
	```bash
	pip install -r requirements.txt
	```

#### Dependencias del sistema (Linux)
Algunos paquetes requieren librerías del sistema. Instala con:
```bash
sudo apt update
sudo apt install gdal-bin libgdal-dev libgeos-dev libspatialindex-dev proj-bin proj-data
```

En Windows, la mayoría de los paquetes se instalan automáticamente desde wheels. Si tienes problemas, consulta la documentación de cada paquete.

### Opción 2: Automática con setup_env.sh

Puedes usar el script automatizado para crear el entorno, instalar dependencias y activar el venv:

```bash
bash setup_env.sh
```

Si "sourceas" el script (`source setup_env.sh`), el entorno queda activado en la shell actual.

El script instala dependencias desde `requirements.txt` y muestra instrucciones finales para ejecutar el procesamiento.

---

## Ejecución del procesamiento de mapas

Con el entorno activado, ejecuta:
```bash
python generar_mapas_tl.py
```
O directamente usando el Python del venv:
```bash
.venv/bin/python generar_mapas_tl.py
```

---

**Notas:**
- Si el script `setup_env.sh` no encuentra `requirements.txt`, deberás crearlo manualmente con las dependencias necesarias.
- Si agregas o actualizas paquetes, actualiza `requirements.txt` con:
  ```bash
  pip freeze > requirements.txt
  ```
