#!/usr/bin/env bash
# setup_env.sh — crea/activa venv, instala deps en el venv y (si es "sourced") deja el venv activo.
set -euo pipefail

VENV_DIR=".venv"

# Detectar si el script está siendo "sourced" (Bash/Zsh)
is_sourced() {
  # Bash
  if [ -n "${BASH_SOURCE-}" ] && [ "${BASH_SOURCE[0]}" != "$0" ]; then
    return 0
  fi
  # Zsh
  if [ -n "${ZSH_EVAL_CONTEXT-}" ] && [[ "$ZSH_EVAL_CONTEXT" == *:file ]]; then
    return 0
  fi
  return 1
}

# 1) Python base
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "❌ No se encontró python3 ni python en el PATH."
  exit 1
fi

# 2) Crear venv si falta
if [ ! -d "$VENV_DIR" ]; then
  echo "▶ Creando entorno virtual en $VENV_DIR ..."
  "$PY" -m venv "$VENV_DIR"
else
  echo "✓ Entorno virtual ya existe en $VENV_DIR"
fi

# Rutas explícitas del venv
VPY="$VENV_DIR/bin/python"
VPIP="$VENV_DIR/bin/pip"

if [ ! -x "$VPY" ]; then
  echo "❌ No se encontró $VPY"
  exit 1
fi

# 3) (Opcional) Activar venv en ESTA shell si el script fue sourced
if is_sourced; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  echo "✓ venv ACTIVADO en esta shell: $(python -V) | $(pip -V)"
else
  echo "ℹ️ Ejecutando instalación SIN activar la shell (se usarán binarios del venv por ruta absoluta)."
fi

# 4) Actualizar pip/setuptools/wheel dentro del venv
echo "▶ Actualizando pip/setuptools/wheel en el venv ..."
"$VPY" -m pip install --upgrade pip setuptools wheel

# 5) Instalar dependencias dentro del venv (versiones probadas con Python 3.10.12)
echo "▶ Instalando dependencias del proyecto en el venv ..."
# ...
"$VPIP" install \
  numpy==1.26.4 \
  pandas==2.2.2 \
  scipy==1.13.1 \
  matplotlib==3.9.0 \
  geopandas==0.14.4 \
  shapely==2.0.4 \
  pyproj==3.6.1 \
  rasterio==1.3.10 \
  cartopy==0.22.0 \
  pykrige==1.7.1 \
  rtree==1.3.0 \
  scikit-learn==1.5.1 \
  tqdm==4.66.4
# ...



# 6) requirements.txt desde el venv
echo "▶ Generando requirements.txt ..."
"$VPIP" freeze > requirements.txt
echo "✓ requirements.txt creado."

# 7) Mensaje final
if is_sourced; then
  echo
  echo "🎉 Entorno listo y ACTIVADO en esta shell."
  echo "Podés ejecutar ahora:"
  echo "  python generar_mapas_tl.py"
else
  echo
  echo "🎉 Entorno listo (no quedó activado porque corriste ./setup_env.sh)."
  echo "Para activarlo en tu shell:"
  echo "  source .venv/bin/activate"
  echo
  echo "O ejecutá el script usando directamente el Python del venv (sin activar):"
  echo "  ./.venv/bin/python generar_mapas_tl.py"
fi
