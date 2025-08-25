#!/usr/bin/env bash
set -euo pipefail

# Carpeta de destino (cambiala si querés)
DEST_DIR="/home/pato/Documentos/Black River"

# Si querés conservar los .zip cambiá a "true"
KEEP_ZIPS=false

# Crea carpeta si no existe
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

# ---- URLs Natural Earth ----
declare -A NE_URLS=(
  # 50m - físicos y políticos
  [ne_50m_coastline]="https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip"
  [ne_50m_land]="https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip"
  [ne_50m_ocean]="https://naciscdn.org/naturalearth/50m/physical/ne_50m_ocean.zip"
  [ne_50m_admin_0_boundary_lines_land]="https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_boundary_lines_land.zip"

  # 10m - culturales y físicos
  [ne_10m_populated_places]="https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
  [ne_10m_roads]="https://naciscdn.org/naturalearth/10m/cultural/ne_10m_roads.zip"
  [ne_10m_admin_1_states_provinces]="https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"
  [ne_10m_rivers_lake_centerlines]="https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_lake_centerlines.zip"
)


echo "Descargando y descomprimiendo shapefiles en: $DEST_DIR"
echo

for NAME in "${!NE_URLS[@]}"; do
  ZIP="${NAME}.zip"
  URL="${NE_URLS[$NAME]}"
  OUT_DIR="${NAME}"

  echo "==> ${NAME}"
  echo "   URL: ${URL}"

  # -c reanuda si está parcialmente descargado
  wget -c -O "$ZIP" "$URL"
  # Crea carpeta y descomprime
  mkdir -p "$OUT_DIR"
  unzip -o "$ZIP" -d "$OUT_DIR" >/dev/null

  # Opcional: eliminar zip para ahorrar espacio
  if [ "$KEEP_ZIPS" = false ]; then
    rm -f "$ZIP"
  fi

  echo "   OK: ${OUT_DIR}/"
  echo
done

echo "Listo ✅"
