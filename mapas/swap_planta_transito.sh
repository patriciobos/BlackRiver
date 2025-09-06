#!/usr/bin/env bash
# Intercambia 'gsm-planta_*_map.png' <-> 'gsm-transito_*_map.png'
# Usa marcadores temporales sin "/" para evitar crear subdirectorios.
# Evita choques: si el destino existe, agrega sufijo _dup antes de la extensión.
#
# Uso:
#   ./swap_planta_transito.sh           # carpeta actual
#   ./swap_planta_transito.sh /ruta     # carpeta indicada

set -euo pipefail
shopt -s nullglob

DIR="${1:-.}"

move_safe() {
  local src="$1"
  local dst="$2"
  if [[ -e "$dst" ]]; then
    local ext="${dst##*.}"
    local base="${dst%.*}"
    local dst2="${base}_dup.${ext}"
    echo "⚠️  Destino ya existe, renombrando con sufijo: '$dst2'"
    mv -i -- "$src" "$dst2"
  else
    mv -i -- "$src" "$dst"
  fi
}

# 1) transito -> marcador TMPPLANTA (sin crear subcarpetas)
for f in "$DIR"/gsm-transito_*_map.png; do
  [[ -e "$f" ]] || continue
  base="${f##*/}"
  new="${base/gsm-transito_/gsm-TMPPLANTA_}"
  src="$f"
  dst="$DIR/$new"
  echo "→ (marca) '$base'  →  '$new'"
  move_safe "$src" "$dst"
done

# 2) planta -> marcador TMPTRANSITO
for f in "$DIR"/gsm-planta_*_map.png; do
  [[ -e "$f" ]] || continue
  base="${f##*/}"
  new="${base/gsm-planta_/gsm-TMPTRANSITO_}"
  src="$f"
  dst="$DIR/$new"
  echo "→ (marca) '$base'  →  '$new'"
  move_safe "$src" "$dst"
done

# 3) TMPPLANTA -> planta
for f in "$DIR"/gsm-TMPPLANTA_*_map.png; do
  [[ -e "$f" ]] || continue
  base="${f##*/}"
  new="${base/gsm-TMPPLANTA_/gsm-planta_}"
  src="$f"
  dst="$DIR/$new"
  echo "→ (final) '$base'  →  '$new'"
  move_safe "$src" "$dst"
done

# 4) TMPTRANSITO -> transito
for f in "$DIR"/gsm-TMPTRANSITO_*_map.png; do
  [[ -e "$f" ]] || continue
  base="${f##*/}"
  new="${base/gsm-TMPTRANSITO_/gsm-transito_}"
  src="$f"
  dst="$DIR/$new"
  echo "→ (final) '$base'  →  '$new'"
  move_safe "$src" "$dst"
done

echo "✅ Renombrado completo."

