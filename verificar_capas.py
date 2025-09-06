# -*- coding: utf-8 -*-
"""
Descarga automática de capas de Natural Earth si no están presentes localmente.
Colocar este archivo junto al script principal.

Uso:
    from verificar_capas import verificar_y_descargar_capas
    verificar_y_descargar_capas(shapefiles, carpeta_capas="Capas")
"""

import os
import io
import glob
import shutil
import zipfile
from urllib.request import urlopen, Request

# ---- Bases de Natural Earth (NACIS CDN) -------------------------------------
BASE_10M_PHYS = "https://naciscdn.org/naturalearth/10m/physical/"
BASE_10M_CULT = "https://naciscdn.org/naturalearth/10m/cultural/"
BASE_50M_PHYS = "https://naciscdn.org/naturalearth/50m/physical/"
BASE_50M_CULT = "https://naciscdn.org/naturalearth/50m/cultural/"

# ---- Mapa: nombre_de_capa -> URL del ZIP ------------------------------------
# Notar que agregamos ambas variantes: "provinces" y "provincias" (alias).
CAPAS_URLS = {
    # 50m físico
    "ne_50m_coastline": BASE_50M_PHYS + "ne_50m_coastline.zip",
    "ne_50m_land": BASE_50M_PHYS + "ne_50m_land.zip",
    "ne_50m_ocean": BASE_50M_PHYS + "ne_50m_ocean.zip",

    # 50m cultural
    "ne_50m_admin_0_boundary_lines_land": BASE_50M_CULT + "ne_50m_admin_0_boundary_lines_land.zip",

    # 10m cultural
    "ne_10m_populated_places": BASE_10M_CULT + "ne_10m_populated_places.zip",
    "ne_10m_roads": BASE_10M_CULT + "ne_10m_roads.zip",
    "ne_10m_admin_1_states_provinces": BASE_10M_CULT + "ne_10m_admin_1_states_provinces.zip",
    "ne_10m_admin_1_states_provincias": BASE_10M_CULT + "ne_10m_admin_1_states_provinces.zip",  # alias

    # 10m físico
    "ne_10m_rivers_lake_centerlines": BASE_10M_PHYS + "ne_10m_rivers_lake_centerlines.zip",
}


def _tiene_shp_en_directorio(dir_path: str) -> bool:
    """Devuelve True si en dir_path existe al menos un .shp."""
    if not os.path.isdir(dir_path):
        return False
    return any(glob.glob(os.path.join(dir_path, "*.shp")))


def _descargar_zip(url: str, timeout: int = 60) -> bytes:
    """Descarga el ZIP y devuelve su contenido como bytes."""
    # User-Agent “amigable” por si algún proxy molesta.
    req = Request(url, headers={"User-Agent": "Python-urllib/3 verificar_capas"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _extraer_zip_en_destino(zip_bytes: bytes, destino_dir: str):
    """
    Extrae el ZIP en destino_dir. Si el ZIP trae una carpeta interna, se preserva.
    Si trae archivos sueltos, se dejan en destino_dir.
    """
    os.makedirs(destino_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Extraemos a un directorio temporal para poder reubicar limpio
        tmp_dir = destino_dir + "_tmp_extract"
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        zf.extractall(tmp_dir)

        # Si el ZIP trae una carpeta raíz, movemos su contenido a destino_dir
        contenidos = os.listdir(tmp_dir)
        if len(contenidos) == 1 and os.path.isdir(os.path.join(tmp_dir, contenidos[0])):
            carpeta_interna = os.path.join(tmp_dir, contenidos[0])
            for nombre in os.listdir(carpeta_interna):
                src = os.path.join(carpeta_interna, nombre)
                dst = os.path.join(destino_dir, nombre)
                if os.path.isdir(src):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    shutil.move(src, dst)
                else:
                    if os.path.isfile(dst):
                        os.remove(dst)
                    shutil.move(src, dst)
        else:
            # Archivos sueltos: mover todo al destino
            for nombre in contenidos:
                src = os.path.join(tmp_dir, nombre)
                dst = os.path.join(destino_dir, nombre)
                if os.path.isdir(src):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    shutil.move(src, dst)
                else:
                    if os.path.isfile(dst):
                        os.remove(dst)
                    shutil.move(src, dst)

        shutil.rmtree(tmp_dir)


def _asegurar_capa_disponible(nombre_capa: str, dir_destino: str) -> bool:
    """
    Garantiza que la capa `nombre_capa` exista con al menos un .shp dentro de `dir_destino`.
    Si no está, intenta descargarla desde CAPAS_URLS y extraerla ahí.
    Devuelve True si queda disponible, False si falla.
    """
    # 1) ¿Ya está?
    if _tiene_shp_en_directorio(dir_destino):
        return True

    # 2) Buscar URL conocida (con alias de provincias/provinces por si acaso)
    url = CAPAS_URLS.get(nombre_capa)
    if not url and nombre_capa.endswith("_provincias"):
        url = CAPAS_URLS.get(nombre_capa.replace("_provincias", "_provinces"))

    if not url:
        print(f"⚠️ No tengo URL conocida para '{nombre_capa}'. Saltando descarga.")
        return False

    print(f"⬇️ Descargando {nombre_capa} desde {url} ...")
    try:
        data = _descargar_zip(url)
    except Exception as e:
        print(f"❌ Error descargando {nombre_capa}: {e}")
        return False

    # 3) Extraer
    try:
        _extraer_zip_en_destino(data, dir_destino)
    except Exception as e:
        print(f"❌ Error extrayendo {nombre_capa}: {e}")
        return False

    # 4) Verificar
    if _tiene_shp_en_directorio(dir_destino):
        print(f"✅ {nombre_capa} disponible en {dir_destino}")
        return True
    else:
        print(f"❌ No se encontró .shp de {nombre_capa} tras la extracción.")
        return False


def verificar_y_descargar_capas(shapefiles: dict, carpeta_capas: str = "Capas"):
    """
    Recorre el dict `shapefiles` {nombre_capa: ruta_dir_o_zip_esperado}
    - Si la ruta apunta a un directorio, asegura que contenga el .shp bajándolo si falta.
    - Si la ruta apunta a un .zip, crea (si no existe) una carpeta con el mismo nombre sin .zip,
      descarga y extrae ahí.
    - Si la ruta apunta a un .shp concreto, asegura el directorio contenedor.

    Modifica el filesystem para que, en la próxima corrida, el script encuentre todo local.
    """
    os.makedirs(carpeta_capas, exist_ok=True)

    for nombre_capa, path_config in shapefiles.items():
        # Normalizar destino:
        #  - si es .zip -> usar carpeta sin .zip
        #  - si es .shp -> usar carpeta contenedora
        #  - si no, asumir carpeta
        if path_config.lower().endswith(".zip"):
            dir_destino = path_config[:-4]
        elif path_config.lower().endswith(".shp"):
            dir_destino = os.path.dirname(path_config)
        else:
            dir_destino = path_config

        # Si la ruta no es absoluta, colgar de carpeta_capas
        if not os.path.isabs(dir_destino):
            dir_destino = os.path.join(carpeta_capas, os.path.basename(dir_destino))

        ok = _asegurar_capa_disponible(nombre_capa, dir_destino)
        if not ok:
            print(f"⚠️ No pude asegurar la capa: {nombre_capa}. "
                  f"Podés descargarla manualmente desde Natural Earth.")
