import os
import requests
import zipfile
import glob

def natural_earth_url(nombre):
    urls = {
        "ne_50m_coastline": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/physical/ne_50m_coastline.zip",
        "ne_50m_land": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/physical/ne_50m_land.zip",
        "ne_50m_ocean": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/physical/ne_50m_ocean.zip",
        "ne_50m_admin_0_boundary_lines_land": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/cultural/ne_50m_admin_0_boundary_lines_land.zip",
        "ne_10m_populated_places": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_populated_places.zip",
        "ne_10m_rivers_lake_centerlines": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_rivers_lake_centerlines.zip",
        "ne_10m_roads": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_roads.zip",
        "ne_10m_admin_1_states_provincias": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_1_states_provinces.zip"
    }
    return urls.get(nombre, None)


def verificar_y_descargar_capas(shapefiles, carpeta_capas="Capas"):
    print("Verificando capas...")
    for nombre, ruta in shapefiles.items():
        subdir = os.path.join(carpeta_capas, nombre)
        shp_files = glob.glob(os.path.join(subdir, "*.shp"))
        if not shp_files:
            print(f"❌ Falta la capa: {nombre}. Intentando descargar...")
            url = natural_earth_url(nombre)
            if url is None:
                print(f"No tengo URL para descargar {nombre}")
                continue
            zip_path = os.path.join(carpeta_capas, f"{nombre}.zip")
            # Descargar ZIP
            try:
                r = requests.get(url, timeout=60)
                if r.status_code != 200:
                    print(f"No se pudo descargar {nombre}: status {r.status_code}")
                    continue
                with open(zip_path, "wb") as f:
                    f.write(r.content)
                print(f"Descargado: {zip_path}")
            except Exception as e:
                print(f"Error descargando {nombre}: {e}")
                print(f"⚠️ Continúo sin la capa {nombre}. Puedes descargarla manualmente si lo necesitas.")
                continue
            # Descomprimir
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(subdir)
                print(f"Descomprimido en: {subdir}")
            except Exception as e:
                print(f"Error descomprimiendo {nombre}: {e}")
                print(f"⚠️ Continúo sin la capa {nombre}. Puedes descomprimirla manualmente si lo necesitas.")
                continue
        else:
            print(f"✔️ Capa presente: {nombre}")
    print("Verificación de capas completa.")
