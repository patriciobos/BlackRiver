#!/bin/bash
# Copia los shapefiles de Natural Earth a la estructura que Cartopy espera

SRC="Capas/shapefiles/natural_earth/physical"
DST="Capas/110m_physical"

mkdir -p "$DST"

for base in ne_110m_ocean ne_110m_land ne_110m_coastline; do
    for ext in shp shx dbf prj cpg; do
        if [ -f "$SRC/$base.$ext" ]; then
            cp "$SRC/$base.$ext" "$DST/"
        fi
    done
    # También copia el archivo .xml si existe
    if [ -f "$SRC/$base.xml" ]; then
        cp "$SRC/$base.xml" "$DST/"
    fi
    # También copia el archivo .zip si existe
    if [ -f "$SRC/$base.zip" ]; then
        cp "$SRC/$base.zip" "$DST/"
    fi
    # También copia el archivo .qpj si existe
    if [ -f "$SRC/$base.qpj" ]; then
        cp "$SRC/$base.qpj" "$DST/"
    fi
    # También copia el archivo .sbn y .sbx si existen
    for ext2 in sbn sbx; do
        if [ -f "$SRC/$base.$ext2" ]; then
            cp "$SRC/$base.$ext2" "$DST/"
        fi
    done

done

echo "Shapefiles copiados a $DST."
