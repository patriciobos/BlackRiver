#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from cartopy.io import shapereader
from shapely.ops import unary_union

from concurrent.futures import ProcessPoolExecutor
import shapely.wkb as wkb

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import patheffects
import shapely.geometry as sgeom
import shapely.ops as ops
import fiona
import pandas as pd
from pyproj import Geod
import numpy as np
from verificar_capas import verificar_y_descargar_capas
import matplotlib.patches as mpatches

# -------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------

# Puntos de análisis
puntos = {
    "p1_transito": {"coords": (-41.41972, -64.40194), "color": "red"},
    "p2_planta": {"coords": (-41.11806, -65.09806), "color": "red"},
}
# Archivos shapefile locales
shapefiles = {
    "ne_50m_coastline": "Capas/ne_50m_coastline",
    "ne_50m_land": "Capas/ne_50m_land",
    "ne_50m_ocean": "Capas/ne_50m_ocean",
    "ne_50m_admin_0_boundary_lines_land": "Capas/ne_50m_admin_0_boundary_lines_land",
    "ne_10m_populated_places": "Capas/ne_10m_populated_places",
    "ne_10m_rivers_lake_centerlines": "Capas/ne_10m_rivers_lake_centerlines",
    "ne_10m_roads": "Capas/ne_10m_roads",
    "ne_10m_admin_1_states_provincias": "Capas/ne_10m_admin_1_states_provincias"
}


# Cantidad de puntos intermedios entre el origen y el punto extremo (por dirección)
N_INTERMEDIOS = 10  # podes cambiarlo luego
# Incluir también el origen y el extremo en el CSV
INCLUIR_ORIGEN_Y_EXTREMO = True


# Bounding box (None para automático)
bounding_box = [-63.5, -65.5, -40.5, -42.5]

# Máxima distancia si no hay costa
radio_max_km = 100

# Instancia de Geod
geod = Geod(ellps="WGS84")

# Direcciones en grados (16 puntos cardinales)
direcciones = np.arange(0, 360, 22.5)
nombres_direcciones = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
]
FALLBACK_CIUDADES = [
    # (nombre, lat, lon)
    ("Viedma", -40.813, -63.003),
    ("San Antonio Oeste", -40.731, -64.947),
    ("Puerto Madryn", -42.769, -65.038),
    ("Rawson", -43.299, -65.102),
    ("Comodoro Rivadavia", -45.864, -67.482),
    ("Las Grutas", -40.800, -65.083),
    ("Puerto Pirámides", -42.583, -64.283),
]


# -------------------------------
# FUNCIONES
# -------------------------------
def plot_provincias(ax, extent):
    try:
        shp = resolve_shp(shapefiles["ne_10m_admin_1_states_provincias"])
        with fiona.open(shp) as src:
            for feat in src:
                props = feat["properties"]
                if props.get("adm0_a3") != "ARG":
                    continue
                geom = sgeom.shape(feat["geometry"])
                if not _geom_within_extent(geom, extent):
                    continue
                ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                                  facecolor="none", edgecolor="#888888",
                                  linewidth=0.6, linestyle=":", zorder=2)
    except Exception as e:
        print(f"⚠️ Provincias no disponibles: {e}")

def plot_ciudades(ax, extent, min_pop=None):
    """
    Dibuja solo las ciudades argentinas que estén dentro del bounding box (extent).
    extent = [lon_min, lon_max, lat_min, lat_max]
    min_pop: si se especifica, filtra por población mínima.
    """
    try:
        shp = resolve_shp(shapefiles["ne_10m_populated_places"])
        with fiona.open(shp) as src:
            for feat in src:
                props = feat["properties"]
                if props.get("ADM0NAME") == "Argentina":
                    pop = props.get("POP_MAX") or props.get("POP_EST") or 0
                    if min_pop is not None and pop < min_pop:
                        continue
                    name = props.get("NAME") or props.get("NAMEASCII")
                    geom = sgeom.shape(feat["geometry"])
                    # Obtener lon, lat correctamente
                    if geom.geom_type == "Point":
                        lon, lat = geom.xy[0][0], geom.xy[1][0]
                    elif geom.geom_type == "MultiPoint" and hasattr(geom, "geoms"):
                        lon, lat = geom.geoms[0].xy[0][0], geom.geoms[0].xy[1][0]
                    else:
                        continue
                    # Filtrar por bounding box
                    if extent[0] <= lon <= extent[1] and extent[2] <= lat <= extent[3]:
                        ax.plot(lon, lat, marker='.', color='k', ms=3, transform=ccrs.PlateCarree())
                        ax.text(lon+0.05, lat+0.05, str(name), fontsize=8, transform=ccrs.PlateCarree())
    except Exception as e:
        print(f"⚠️ No pude cargar ciudades desde {shapefiles['ne_10m_populated_places']}: {e}")
        # Fallback manual
        for name, lat, lon in FALLBACK_CIUDADES:
            if extent[0] <= lon <= extent[1] and extent[2] <= lat <= extent[3]:
                ax.plot(lon, lat, marker='.', color='k', ms=3, transform=ccrs.PlateCarree())
                ax.text(lon+0.05, lat+0.05, str(name), fontsize=8, transform=ccrs.PlateCarree())

def resolve_shp(path_like):
    import os, glob
    if path_like.lower().endswith(".shp") and os.path.isfile(path_like):
        return path_like
    if os.path.isdir(path_like):
        candidates = sorted(glob.glob(os.path.join(path_like, "*.shp")))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"No se encontró ningún .shp en: {path_like}")
    if path_like.lower().endswith(".zip"):
        folder = path_like[:-4]
        if os.path.isdir(folder):
            return resolve_shp(folder)
    raise FileNotFoundError(f"No se pudo resolver un .shp válido desde: {path_like}")

def cargar_costa(path_zip_o_dir):
    """Carga geometría de costa desde shapefile local (.shp dentro de una carpeta)."""
    shp_path = resolve_shp(path_zip_o_dir)
    geoms = []
    with fiona.open(shp_path) as src:
        for feat in src:
            geoms.append(sgeom.shape(feat["geometry"]))
    return ops.unary_union(geoms)

def _worker_direccion(args):
    nombre, lat, lon, az, dir_name, max_km, costa_wkb, n_inter, incluir_endpoints = args
    geod_local = Geod(ellps="WGS84")
    costa_local = wkb.loads(costa_wkb)

    # 1) calcular extremo (cortado por costa o por 100 km)
    linea = geodesic_line(lat, lon, az, max_km, step_km=0.5, geod_obj=geod_local)
    p = cortar_en_costa(linea, costa_local)
    if p is not None:
        lat1, lon1 = p.y, p.x
    else:
        lon1, lat1, _ = geod_local.fwd(lon, lat, az, max_km * 1000.0)

    # 2) muestrear intermedios (y opcionalmente endpoints)
    puntos = samplear_geodesica(lat, lon, lat1, lon1, n_inter, incluir_endpoints, geod_obj=geod_local)

    filas = []
    for (lati, loni, frac, dist_km) in puntos:
        filas.append({
            "punto": nombre,
            "direccion": dir_name,
            "lat": lati,
            "lon": loni,
            "fraccion": round(frac, 6),          # 0..1
            "dist_km": round(dist_km, 3),        # desde el origen
            "sedimento": ""
        })
    return filas


def calcular_punto_en_direccion(lat, lon, azimut, max_km, costa):
    """
    Calcula el punto final de una geodésica desde (lat, lon) en dirección azimut.
    Corta en la primera intersección con la costa o en max_km si no hay.
    """
    npts = 200  # densidad de muestreo
    lonlats = geod.npts(lon, lat, lon + np.cos(np.radians(azimut)), lat + np.sin(np.radians(azimut)), npts)
    # Agrego el punto inicial
    lonlats = [(lon, lat)] + lonlats

    line = sgeom.LineString(lonlats)

    inter = line.intersection(costa)
    if not inter.is_empty:
        if inter.geom_type == "MultiPoint":
            punto = list(inter)[0]
        elif inter.geom_type == "Point":
            punto = inter
        else:
            return geod.fwd(lon, lat, azimut, max_km * 1000)[:2][::-1]  # lat, lon
        return punto.y, punto.x
    else:
        lon2, lat2, _ = geod.fwd(lon, lat, azimut, max_km * 1000)
        return lat2, lon2

def calcular_direcciones_por_punto(nombre, lat, lon, costa, executor=None):
    tareas = []
    costa_wkb = costa.wkb
    for az, dir_name in zip(direcciones, nombres_direcciones):
        tareas.append((nombre, lat, lon, float(az), dir_name,
                       float(radio_max_km), costa_wkb, int(N_INTERMEDIOS), bool(INCLUIR_ORIGEN_Y_EXTREMO)))

    if executor is None:
        out = []
        for t in tareas:
            filas = _worker_direccion(t)
            out.extend(filas)
        return out
    else:
        out = []
        for filas in executor.map(_worker_direccion, tareas, chunksize=4):
            out.extend(filas)
        return out

def samplear_geodesica(lat0, lon0, lat1, lon1, n_intermedios, incluir_endpoints=True, geod_obj=None):
    """
    Devuelve una lista de puntos a lo largo de la geodésica entre (lat0,lon0) y (lat1,lon1).
    Si incluir_endpoints=True, incluye origen (f=0) y extremo (f=1).
    Con n_intermedios=10: genera 10 puntos estrictamente entre 0 y 1.
    """
    geod_local = geod_obj if geod_obj is not None else geod

    # Distancia total y azimut desde origen a extremo
    az12, az21, dist_m = geod_local.inv(lon0, lat0, lon1, lat1)

    # Fracciones
    if n_intermedios < 0:
        n_intermedios = 0

    fracs = []
    if incluir_endpoints:
        fracs.append(0.0)

    if n_intermedios > 0:
        step = 1.0 / (n_intermedios + 1)
        for k in range(1, n_intermedios + 1):
            fracs.append(k * step)

    if incluir_endpoints:
        fracs.append(1.0)

    # Construcción de puntos
    out = []
    for f in fracs:
        d = f * dist_m
        lonf, latf, _ = geod_local.fwd(lon0, lat0, az12, d)
        out.append((latf, lonf, f, d / 1000.0))  # (lat, lon, fracción, distancia_km)

    return out

def dibujar_rosa_referencia(fig, anchor=(0.15, 0.15), size=0.18):
    axr = fig.add_axes([anchor[0], anchor[1], size, size])
    axr.set_aspect('equal'); axr.axis('off')

    R = 1.0
    for i, name in enumerate(nombres_direcciones):
        az = i * 22.5  # 0..337.5
        # 0° hacia arriba (y+), convertimos a rad inverso y+
        theta = np.deg2rad(90 - az)
        x2 = R * np.cos(theta)
        y2 = R * np.sin(theta)
        axr.plot([0, x2], [0, y2], lw=1)
        # etiqueta un poquito más afuera
        axr.text(1.12*np.cos(theta), 1.12*np.sin(theta), name,
                 ha='center', va='center', fontsize=7)
    # círculo externo
    circ = mpatches.Circle((0,0), R, fill=False, lw=1)
    axr.add_patch(circ)
    axr.text(0, 0, "16-pt", ha='center', va='center', fontsize=8)


def cargar_geoms_linea(path_dir):
    shp = resolve_shp(path_dir)
    geoms = []
    with fiona.open(shp) as src:
        for f in src:
            g = sgeom.shape(f["geometry"])
            if not g.is_empty:
                geoms.append(g)
    return unary_union(geoms)

def cargar_geoms_poligono(path_dir):
    shp = resolve_shp(path_dir)
    geoms = []
    with fiona.open(shp) as src:
        for f in src:
            g = sgeom.shape(f["geometry"])
            if not g.is_empty:
                geoms.append(g)
    return unary_union(geoms)


def exportar_csv(nombre, coords):
    """Guarda las coordenadas en un CSV por punto, corrigiendo dist_km si está en metros y asegurando formato decimal estándar."""
    df = pd.DataFrame(coords)
    if "dist_km" in df.columns:
        # Corrige valores sospechosos (>500 km) asumiendo que están en metros
        df["dist_km"] = df["dist_km"].apply(lambda x: x/1000 if x > 500 else x)
    df.to_csv(f"{nombre}_direcciones.csv", index=False, float_format='%.6f', encoding='utf-8', sep=',')

def _geom_within_extent(geom, extent):
    """Devuelve True si la geometría toca el bbox [lon_min, lon_max, lat_min, lat_max]."""
    lon_min, lon_max, lat_min, lat_max = extent
    bbox = sgeom.box(lon_min, lat_min, lon_max, lat_max)
    return geom.intersects(bbox)

def plot_rios(ax, extent, min_scalerank=5):
    """
    Dibuja ríos (centerlines) dentro del bbox.
    min_scalerank: menor = más importante. Usa <= para mostrar principales.
    """
    try:
        shp = resolve_shp(shapefiles["ne_10m_rivers_lake_centerlines"])
        with fiona.open(shp) as src:
            for feat in src:
                props = feat["properties"]
                scalerank = props.get("scalerank", 10)  # 0..10 aprox.
                geom = sgeom.shape(feat["geometry"])
                if geom.is_empty:
                    continue
                if not _geom_within_extent(geom, extent):
                    continue
                # Ríos principales
                if scalerank <= min_scalerank:
                    ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                                      facecolor="none", edgecolor="#2878B5",
                                      linewidth=1.0, zorder=2)
    except Exception as e:
        print(f"⚠️ No pude cargar ríos: {e}")

def plot_rutas(ax, extent, allowed_types=("Major Highway", "Road"), lw_major=1.2, lw_minor=0.8):
    """
    Dibuja rutas principales dentro del bbox.
    allowed_types: filtra por 'type' en NE (p.ej. 'Major Highway', 'Road').
    """
    try:
        shp = resolve_shp(shapefiles["ne_10m_roads"])
        with fiona.open(shp) as src:
            for feat in src:
                props = feat["properties"]
                rtype = props.get("type", "")
                geom = sgeom.shape(feat["geometry"])
                if geom.is_empty:
                    continue
                if not _geom_within_extent(geom, extent):
                    continue

                if rtype in allowed_types:
                    lw = lw_major if "Major" in rtype else lw_minor
                    ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                                      facecolor="none", edgecolor="#D97706",  # naranja quemado
                                      linewidth=lw, zorder=3, linestyle="-")
    except Exception as e:
        print(f"⚠️ No pude cargar rutas: {e}")


def graficar_mapa(puntos, resultados, costa):
    """Grafica el mapa principal y el inset planisferio."""
    # Determinar bounding box
    if bounding_box:
        extent = bounding_box
    else:
        lats = [p["coords"][0] for p in puntos.values()]
        lons = [p["coords"][1] for p in puntos.values()]
        margin = 1.0
        extent = [min(lons)-margin, max(lons)+margin,
                min(lats)-margin, max(lats)+margin]
    figsize = calcular_figsize(extent, base_height=8)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="lightblue")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="lightgray")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linestyle="--", edgecolor="black")
    gl = ax.gridlines(draw_labels=True, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    plot_rios(ax, extent, min_scalerank=5)
    plot_rutas(ax, extent, allowed_types=("Major Highway","Road"))
    plot_ciudades(ax, extent, min_pop=20000)
    plot_provincias(ax, extent)
    ax.text(-66, -41, "ARGENTINA", fontsize=14, weight="bold", color="gray",
            transform=ccrs.PlateCarree(),
            path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])
    for nombre, info in puntos.items():
        lat, lon = info["coords"]
        df = resultados[nombre]
        for dir_name in nombres_direcciones:
            sub = df[df["direccion"] == dir_name].sort_values("fraccion")
            ax.plot([lon] + sub["lon"].tolist(),
                    [lat] + sub["lat"].tolist(),
                    color="black", linewidth=0.9,
                    transform=ccrs.Geodetic(), zorder=2)
            ax.scatter(sub["lon"], sub["lat"],
                    s=30,
                    facecolor="yellow",
                    edgecolor="black",
                    transform=ccrs.PlateCarree(),
                    zorder=3)
            if not sub.empty:
                lat_end, lon_end = sub.iloc[-1][["lat", "lon"]]
                ax.plot(lon_end, lat_end, marker="o", markersize=8,
                        markeredgecolor="red", markerfacecolor="white",
                        transform=ccrs.PlateCarree(), zorder=4)
    # Inset Atlántico Sur
    sub_ax = fig.add_axes((0.70, 0.70, 0.25, 0.25), projection=ccrs.PlateCarree())
    atlantico_sur_extent = [-70, -40, -60, -30]
    sub_ax.set_extent(atlantico_sur_extent, crs=ccrs.PlateCarree())
    sub_ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="lightblue")
    sub_ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="lightgray")
    sub_ax.coastlines()
    extent_inset = ax.get_extent()
    rect = sgeom.box(extent_inset[0], extent_inset[2], extent_inset[1], extent_inset[3])
    sub_ax.add_geometries([rect], crs=ccrs.PlateCarree(), facecolor="none", edgecolor="red", zorder=20)
    dibujar_rosa_referencia(fig)
    out_png = "mapa_golfo.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Mapa guardado en: {out_png}")

    for nombre, info in puntos.items():
        lat, lon = info["coords"]
        df = resultados[nombre]
        for dir_name in nombres_direcciones:
            sub = df[df["direccion"] == dir_name].sort_values("fraccion")
            # Línea pasando por todos los intermedios
            ax.plot([lon] + sub["lon"].tolist(), [lat] + sub["lat"].tolist(),
                    color="black", linewidth=0.9, transform=ccrs.Geodetic(), zorder=3)
            # Marcadores visibles para intermedios y extremo
            ax.plot(sub["lon"], sub["lat"],
                    marker="o", markersize=5,
                    markeredgecolor="black", markerfacecolor="yellow",
                    linestyle="None", transform=ccrs.PlateCarree(), zorder=4)
            # Marcadores sobre la recta (tantos como puntos en el csv para esa dirección)
            for idx, row in sub.iterrows():
                ax.plot(row["lon"], row["lat"], marker="o", markersize=7,
                        markeredgecolor="blue", markerfacecolor="cyan",
                        linestyle="None", transform=ccrs.PlateCarree(), zorder=5)

def calcular_figsize(extent, base_height=8):
    """
    Ajusta el tamaño de la figura proporcional al bounding box.
    extent = [lon_min, lon_max, lat_min, lat_max]
    """
    lon_min, lon_max, lat_min, lat_max = extent
    width = lon_max - lon_min
    height = lat_max - lat_min
    aspect = width / height if height != 0 else 1
    return (base_height * aspect, base_height)


def geodesic_line(lat, lon, azimut_deg, max_km, step_km=0.5, geod_obj=None):
    geod_local = geod_obj if geod_obj is not None else geod
    dists_m = np.linspace(0, max_km*1000.0, int(max_km/step_km)+1)
    lons, lats = [], []
    for d in dists_m:
        lon2, lat2, _ = geod_local.fwd(lon, lat, azimut_deg, d)
        lons.append(lon2); lats.append(lat2)
    return sgeom.LineString(zip(lons, lats))


def cortar_en_costa(linea, costa):
    inter = linea.intersection(costa)
    if inter.is_empty:
        return None
    # Puede devolver MultiPoint, GeometryCollection, LineString, etc.
    # Nos quedamos con el/los puntos de intersección y tomamos el más cercano al origen de la línea.
    pts = []
    if inter.geom_type == "Point":
        pts = [inter]
    elif inter.geom_type in ("MultiPoint", "GeometryCollection"):
        pts = [g for g in inter.geoms if g.geom_type == "Point"]
    elif inter.geom_type in ("LineString", "MultiLineString"):
        # Si toca a lo largo (colineal), tomamos el primer vértice proyectado
        if inter.geom_type == "LineString":
            pts = [sgeom.Point(inter.coords[0])]
        else:
            pts = [sgeom.Point(list(inter.geoms)[0].coords[0])]
    if not pts:
        return None
    # Elegimos el punto con menor distancia proyectada a lo largo de la línea
    dmin = None
    pmin = None
    for p in pts:
        d = linea.project(p)
        if dmin is None or d < dmin:
            dmin = d; pmin = p
    return pmin

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    faltantes = []
    verificar_y_descargar_capas(shapefiles, carpeta_capas="Capas")
    # Cargar geometrías
    try:
        costa = cargar_geoms_linea(shapefiles["ne_50m_coastline"])
    except Exception as e:
        print(f"⚠️ No se pudo cargar la capa de costa: {e}")
        costa = None
        faltantes.append("ne_50m_coastline")
    try:
        tierra = cargar_geoms_poligono(shapefiles["ne_50m_land"])
    except Exception as e:
        print(f"⚠️ No se pudo cargar la capa de tierra: {e}")
        tierra = None
        faltantes.append("ne_50m_land")
    resultados = {}
    max_workers = os.cpu_count() or 1
    step_km = 2.0  # separación de puntos sobre cada track

    def calcular_direcciones_por_punto_km(nombre, lat, lon, costa, executor=None, step_km=2.0):
        tareas = []
        costa_wkb = costa.wkb
        for az, dir_name in zip(direcciones, nombres_direcciones):
            tareas.append((nombre, lat, lon, float(az), dir_name,
                           float(radio_max_km), costa_wkb, int(radio_max_km/step_km)-1, True))
        if executor is None:
            out = []
            for t in tareas:
                filas = _worker_direccion(t)
                out.extend(filas)
            return out
        else:
            out = []
            for filas in executor.map(_worker_direccion, tareas, chunksize=4):
                out.extend(filas)
            return out

    def exportar_kml(nombre, df):
        from simplekml import Kml, Style
        kml = Kml()
        style_line = Style()
        style_line.linestyle.width = 2
        style_line.linestyle.color = 'ff0000ff'  # rojo
        style_point = Style()
        style_point.iconstyle.color = 'ff00ffff'  # amarillo
        for dir_name in nombres_direcciones:
            sub = df[df["direccion"] == dir_name].sort_values("fraccion")
            coords = [(row["lon"], row["lat"]) for _, row in sub.iterrows()]
            if len(coords) > 1:
                ls = kml.newlinestring(name=f"{nombre}_{dir_name}", coords=coords)
                ls.style = style_line
            for lon, lat in coords:
                p = kml.newpoint(coords=[(lon, lat)])
                p.style = style_point
        kml.save(f"{nombre}_tracks.kml")

    # Generar mapas y KML por punto
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for nombre, info in puntos.items():
            lat, lon = info["coords"]
            coords = calcular_direcciones_por_punto_km(nombre, lat, lon, costa, executor=ex, step_km=step_km)
            exportar_csv(nombre, coords)
            df = pd.DataFrame(coords)
            resultados = {nombre: df}
            graficar_mapa({nombre: info}, resultados, costa)
            exportar_kml(nombre, df)

    # Al final del script
    if faltantes:
        print("\n⚠️ Capas faltantes (descarga manual recomendada):")
        for capa in faltantes:
            print(f"  - {capa} (ver https://www.naturalearthdata.com)")
    else:
        print("\nTodas las capas principales fueron cargadas correctamente.")

