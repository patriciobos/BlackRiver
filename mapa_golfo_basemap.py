#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Circle

from pyproj import Geod
from verificar_capas import verificar_y_descargar_capas
from mpl_toolkits.basemap import Basemap

# -------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------

# Puntos de análisis
puntos = {
    "p1_transito": {"coords": (-41.41972, -64.40194), "color": "red"},
    "p2_planta": {"coords": (-41.11806, -65.09806), "color": "red"},
}

# Natural Earth locales (rutas relativas dentro de Capas/)
shapefiles = {
    "ne_50m_land": "Capas/ne_50m_land",
    "ne_50m_coastline": "Capas/ne_50m_coastline",  # no la usamos para cortar, pero sí podés dibujar si querés
    "ne_50m_ocean": "Capas/ne_50m_ocean",
    "ne_50m_admin_0_boundary_lines_land": "Capas/ne_50m_admin_0_boundary_lines_land",
    "ne_10m_populated_places": "Capas/ne_10m_populated_places",
    "ne_10m_rivers_lake_centerlines": "Capas/ne_10m_rivers_lake_centerlines",
    "ne_10m_roads": "Capas/ne_10m_roads",
    "ne_10m_admin_1_states_provincias": "Capas/ne_10m_admin_1_states_provincias",  # alias a provinces
}

# Bounding box CORRECTO: [lon_min, lon_max, lat_min, lat_max]
bounding_box = [-65.5, -63.5, -42.5, -40.5]

# Muestreo de direcciones
radio_max_km = 100
N_INTERMEDIOS = 10
INCLUIR_ORIGEN_Y_EXTREMO = True
step_km_tracks = 2.0  # separación entre puntos de cada track (para CSV)

# Geodesia
geod = Geod(ellps="WGS84")
direcciones = np.arange(0, 360, 22.5)
nombres_direcciones = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
]

warnings.filterwarnings("ignore")

# -------------------------------
# HELPERS
# -------------------------------
def _normalize_extent(ext):
    lon0, lon1, lat0, lat1 = ext
    return [min(lon0, lon1), max(lon0, lon1), min(lat0, lat1), max(lat0, lat1)]

def resolve_shp(path_like):
    """Devuelve el primer .shp si path_like es carpeta; si es .shp lo devuelve; si es .zip usa carpeta sin .zip."""
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

def shp_prefix_without_ext(path_like):
    """Basemap.readshapefile pide el path SIN extensión; devolvemos el prefijo del .shp."""
    shp = resolve_shp(path_like)
    return os.path.splitext(shp)[0]

# -------------------------------
# GEODÉSICAS (sin Shapely)
# -------------------------------
def first_coast_crossing_with_basemap(m, lat, lon, az_deg, max_km, coarse_step_km=0.5, refine_iters=12):
    """
    Avanza a lo largo de la geodésica (lat,lon,az) y detecta el primer cruce tierra-mar usando Basemap.is_land().
    Devuelve (lat_cruce, lon_cruce). Si no cruza, devuelve el punto a max_km.
    """
    # Basemap en 'cyl' -> is_land espera (lon, lat)
    start_is_land = bool(m.is_land(lon, lat))

    # Búsqueda lineal gruesa
    d = coarse_step_km
    while d <= max_km + 1e-9:
        lon2, lat2, _ = geod.fwd(lon, lat, az_deg, d * 1000.0)
        curr_is_land = bool(m.is_land(lon2, lat2))
        if curr_is_land != start_is_land:
            # Refinamiento binario entre d - step y d
            lo = d - coarse_step_km
            hi = d
            for _ in range(refine_iters):
                mid = 0.5 * (lo + hi)
                lonm, latm, _ = geod.fwd(lon, lat, az_deg, mid * 1000.0)
                mid_is_land = bool(m.is_land(lonm, latm))
                if mid_is_land == start_is_land:
                    lo = mid
                else:
                    hi = mid
            lonf, latf, _ = geod.fwd(lon, lat, az_deg, hi * 1000.0)
            return latf, lonf
        d += coarse_step_km

    # No hubo cruce: devolver extremo a max_km
    lonf, latf, _ = geod.fwd(lon, lat, az_deg, max_km * 1000.0)
    return latf, lonf

def samplear_geodesica(lat0, lon0, lat1, lon1, n_intermedios, incluir_endpoints=True):
    """
    Devuelve una lista de puntos a lo largo de la geodésica entre (lat0,lon0) y (lat1,lon1).
    Si incluir_endpoints=True, incluye origen (f=0) y extremo (f=1).
    Con n_intermedios=10: genera 10 puntos estrictamente entre 0 y 1.
    """
    az12, az21, dist_m = geod.inv(lon0, lat0, lon1, lat1)

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

    out = []
    for f in fracs:
        d = f * dist_m
        lonf, latf, _ = geod.fwd(lon0, lat0, az12, d)
        out.append((latf, lonf, f, d / 1000.0))  # (lat, lon, fracción, dist_km)
    return out

def calcular_direcciones_por_punto_km_basemap(nombre, lat, lon, m, step_km=2.0):
    """Genera filas (lista de dict) con puntos muestreados hasta la costa usando Basemap.is_land()."""
    filas_tot = []
    for az, dir_name in zip(direcciones, nombres_direcciones):
        lat1, lon1 = first_coast_crossing_with_basemap(m, lat, lon, float(az), radio_max_km,
                                                       coarse_step_km=0.5, refine_iters=12)
        n_inter = int(max(1, radio_max_km / step_km) - 1)
        puntos = samplear_geodesica(lat, lon, lat1, lon1, n_inter, incluir_endpoints=True)
        for (lati, loni, frac, dist_km) in puntos:
            filas_tot.append({
                "punto": nombre,
                "direccion": dir_name,
                "lat": lati,
                "lon": loni,
                "fraccion": round(frac, 6),
                "dist_km": round(dist_km, 3),
                "sedimento": ""
            })
    return filas_tot

# -------------------------------
# DIBUJO CON BASEMAP + NE
# -------------------------------
def _crear_mapa(extent):
    """Basemap en proyección cilíndrica, con land/sea y capas base listas para is_land()."""
    lon_min, lon_max, lat_min, lat_max = _normalize_extent(extent)
    m = Basemap(
        projection='cyl',
        llcrnrlon=lon_min, urcrnrlon=lon_max,
        llcrnrlat=lat_min, urcrnrlat=lat_max,
        resolution='i'  # 'c','l','i','h','f'
    )
    # Fondo mar y tierra (esto inicializa polígonos internos para is_land)
    m.drawmapboundary(fill_color='lightblue', linewidth=0.8)
    m.fillcontinents(color='lightgray', lake_color='lightblue')
    # Fronteras políticas y costa
    m.drawcountries(linewidth=0.6, linestyle='--', color='black')
    m.drawstates(linewidth=0.4, linestyle=':', color='gray')
    m.drawcoastlines(linewidth=0.6, color='black')
    # Grilla
    m.drawparallels(np.arange(-90, 91, 1), labels=[1,0,0,0], dashes=[2,2], linewidth=0.3)
    m.drawmeridians(np.arange(-180, 181, 1), labels=[0,0,0,1], dashes=[2,2], linewidth=0.3)
    return m

def _basemap_plot_shp_lines(m, shp_path_prefix, filter_func=None, edgecolor='k', linewidth=1.0, zorder=3, linestyle='-'):
    name = os.path.basename(shp_path_prefix).replace("-", "_")
    m.readshapefile(shp_path_prefix, name, drawbounds=False)
    shp = getattr(m, name)
    info = getattr(m, f"{name}_info")
    for geom, meta in zip(shp, info):
        if filter_func and not filter_func(meta):
            continue
        xs, ys = zip(*geom)
        plt.plot(xs, ys, linestyle=linestyle, linewidth=linewidth, color=edgecolor, zorder=zorder)

def _basemap_plot_points_from_shp(m, shp_path_prefix, filter_func=None, ms=3, zorder=4):
    name = os.path.basename(shp_path_prefix).replace("-", "_")
    m.readshapefile(shp_path_prefix, name, drawbounds=False)
    shp = getattr(m, name)
    info = getattr(m, f"{name}_info")
    for geom, meta in zip(shp, info):
        if filter_func and not filter_func(meta):
            continue
        if len(geom) == 1:
            x, y = geom[0]
            plt.plot(x, y, marker='.', color='k', ms=ms, zorder=zorder)
        else:
            xs, ys = zip(*geom)
            plt.plot(xs, ys, linestyle='None', marker='.', color='k', ms=ms, zorder=zorder)

def plot_provincias_basemap(m):
    prefix = shp_prefix_without_ext(shapefiles["ne_10m_admin_1_states_provincias"])
    def filt(meta):
        return (meta.get('adm0_a3') == 'ARG')
    _basemap_plot_shp_lines(m, prefix, filter_func=filt, edgecolor='#888888', linewidth=0.6, zorder=3, linestyle=':')

def plot_rios_basemap(m, min_scalerank=5):
    prefix = shp_prefix_without_ext(shapefiles["ne_10m_rivers_lake_centerlines"])
    def filt(meta):
        scalerank = meta.get('scalerank', 10)
        return scalerank <= min_scalerank
    _basemap_plot_shp_lines(m, prefix, filter_func=filt, edgecolor='#2878B5', linewidth=1.0, zorder=3)

def plot_rutas_basemap(m, allowed_types=("Major Highway", "Road"), lw_major=1.2, lw_minor=0.8):
    prefix = shp_prefix_without_ext(shapefiles["ne_10m_roads"])
    name = os.path.basename(prefix).replace("-", "_")
    m.readshapefile(prefix, name, drawbounds=False)
    shp = getattr(m, name)
    info = getattr(m, f"{name}_info")
    for geom, meta in zip(shp, info):
        rtype = meta.get('type', '')
        if rtype in allowed_types:
            lw = lw_major if "Major" in rtype else lw_minor
            xs, ys = zip(*geom)
            plt.plot(xs, ys, linestyle='-', linewidth=lw, color='#D97706', zorder=3.2)

def plot_ciudades_basemap(m, min_pop=20000):
    """
    Dibuja ciudades argentinas desde NE 10m populated places.
    Soporta geometrías como (lon,lat) y listas de (lon,lat).
    Filtra por bounding_box para evitar dibujar fuera del área.
    """
    prefix = shp_prefix_without_ext(shapefiles["ne_10m_populated_places"])
    name = os.path.basename(prefix).replace("-", "_")
    m.readshapefile(prefix, name, drawbounds=False)
    shp = getattr(m, name)
    info = getattr(m, f"{name}_info")

    ext = _normalize_extent(bounding_box)

    def draw_point_if_in_extent(x, y, meta):
        if ext[0] <= x <= ext[1] and ext[2] <= y <= ext[3]:
            plt.plot(x, y, marker='.', color='k', ms=3, zorder=4)
            name_txt = meta.get('NAME') or meta.get('NAMEASCII')
            if name_txt:
                plt.text(x+0.05, y+0.05, str(name_txt), fontsize=8, zorder=4.1)

    for geom, meta in zip(shp, info):
        if meta.get('ADM0NAME') != 'Argentina':
            continue
        pop = meta.get('POP_MAX') or meta.get('POP_EST') or 0
        if min_pop is not None and pop < min_pop:
            continue

        # Caso 1: un solo punto como (lon, lat)
        if isinstance(geom, (list, tuple, np.ndarray)):
            # (lon, lat) -> dos escalares
            if len(geom) == 2 and np.isscalar(geom[0]) and np.isscalar(geom[1]):
                x, y = float(geom[0]), float(geom[1])
                draw_point_if_in_extent(x, y, meta)
                continue

            # Caso 2: multipunto [(lon,lat), (lon,lat), ...]
            try:
                for pt in geom:
                    if isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2:
                        x, y = float(pt[0]), float(pt[1])
                        draw_point_if_in_extent(x, y, meta)
            except Exception:
                # Silencioso si alguna geometría rara se cuela
                pass
        # Otros tipos se ignoran

# -------------------------------
# GRAFICADO ALTO NIVEL
# -------------------------------
def dibujar_rosa_referencia_basemap(ax, anchor=(0.15, 0.15), size=0.18):
    axr = ax.figure.add_axes([anchor[0], anchor[1], size, size])
    axr.set_aspect('equal'); axr.axis('off')
    R = 1.0
    for i, name in enumerate(nombres_direcciones):
        az = i * 22.5
        theta = np.deg2rad(90 - az)
        x2 = R * np.cos(theta)
        y2 = R * np.sin(theta)
        axr.plot([0, x2], [0, y2], lw=1)
        axr.text(1.12*np.cos(theta), 1.12*np.sin(theta), name,
                 ha='center', va='center', fontsize=7)
    circ = Circle((0,0), R, fill=False, lw=1)
    axr.add_patch(circ)
    axr.text(0, 0, "16-pt", ha='center', va='center', fontsize=8)

def calcular_figsize(extent, base_height=8):
    lon_min, lon_max, lat_min, lat_max = _normalize_extent(extent)
    width = lon_max - lon_min
    height = lat_max - lat_min
    aspect = width / height if height != 0 else 1
    return (base_height * aspect, base_height)

def graficar_mapa_basemap(puntos_dict, resultados, extent, out_png, m=None):
    figsize = calcular_figsize(extent, base_height=8)
    fig, ax = plt.subplots(figsize=figsize)
    if m is None:
        m = _crear_mapa(extent)
    else:
        # Redibujar fondo básico para el PNG actual
        m.drawmapboundary(fill_color='lightblue', linewidth=0.8)
        m.fillcontinents(color='lightgray', lake_color='lightblue')
        m.drawcountries(linewidth=0.6, linestyle='--', color='black')
        m.drawstates(linewidth=0.4, linestyle=':', color='gray')
        m.drawcoastlines(linewidth=0.6, color='black')

    # Overlays NE
    plot_rios_basemap(m, min_scalerank=5)
    plot_rutas_basemap(m, allowed_types=("Major Highway","Road"))
    plot_ciudades_basemap(m, min_pop=20000)
    plot_provincias_basemap(m)

    # Label país
    plt.text(-66, -41, "ARGENTINA", fontsize=14, weight="bold", color="gray",
             path_effects=[patheffects.withStroke(linewidth=3, foreground="white")], zorder=2)

    # Tracks
    for nombre, info in puntos_dict.items():
        lat0, lon0 = info["coords"]
        df = resultados[nombre]
        for dir_name in nombres_direcciones:
            sub = df[df["direccion"] == dir_name].sort_values("fraccion")
            xs = [lon0] + sub["lon"].tolist()
            ys = [lat0] + sub["lat"].tolist()
            plt.plot(xs, ys, color='black', linewidth=0.9, zorder=2)
            plt.scatter(sub["lon"], sub["lat"], s=30, facecolor="yellow", edgecolor="black", zorder=3)
            if not sub.empty:
                lat_end, lon_end = sub.iloc[-1][["lat", "lon"]]
                plt.plot(lon_end, lat_end, marker="o", markersize=8,
                         markeredgecolor="red", markerfacecolor="white", zorder=4)

    dibujar_rosa_referencia_basemap(ax)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Mapa guardado en: {out_png}")

# -------------------------------
# EXPORTACIÓN
# -------------------------------
def exportar_csv(nombre, coords):
    df = pd.DataFrame(coords)
    if "dist_km" in df.columns:
        df["dist_km"] = df["dist_km"].apply(lambda x: x/1000 if x > 500 else x)
    df.to_csv(f"{nombre}_direcciones.csv", index=False, float_format='%.6f', encoding='utf-8', sep=',')

def exportar_kml(nombre, df):
    try:
        from simplekml import Kml, Style
        kml = Kml()
        style_line = Style(); style_line.linestyle.width = 2; style_line.linestyle.color = 'ff0000ff'
        style_point = Style(); style_point.iconstyle.color = 'ff00ffff'
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
        print(f"KML guardado con simplekml: {nombre}_tracks.kml")
        return
    except ModuleNotFoundError:
        print("simplekml no instalado; generando KML básico...")

    def _kml_escape(t):
        return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    line_color_abgr = "ff0000ff"
    point_color_abgr = "ff00ffff"
    partes = []
    partes.append('<?xml version="1.0" encoding="UTF-8"?>')
    partes.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    partes.append("<Document>")
    partes.append(f"""
    <Style id="lineStyle">
      <LineStyle><color>{line_color_abgr}</color><width>2</width></LineStyle>
    </Style>
    <Style id="ptStyle">
      <IconStyle><color>{point_color_abgr}</color><scale>1.2</scale>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
      </IconStyle>
    </Style>
    """.strip())
    for dir_name in nombres_direcciones:
        sub = df[df["direccion"] == dir_name].sort_values("fraccion")
        coords = [(row["lon"], row["lat"]) for _, row in sub.iterrows()]
        if len(coords) > 1:
            coord_str = " ".join([f"{lon},{lat},0" for lon, lat in coords])
            partes.append("<Placemark>")
            partes.append(f"<name>{_kml_escape(nombre)}_{_kml_escape(dir_name)}</name>")
            partes.append('<styleUrl>#lineStyle</styleUrl>')
            partes.append("<LineString><tessellate>1</tessellate><coordinates>")
            partes.append(coord_str)
            partes.append("</coordinates></LineString>")
            partes.append("</Placemark>")
        for lon, lat in coords:
            partes.append("<Placemark>")
            partes.append(f"<name>{_kml_escape(nombre)} pt</name>")
            partes.append('<styleUrl>#ptStyle</styleUrl>')
            partes.append("<Point><coordinates>")
            partes.append(f"{lon},{lat},0")
            partes.append("</coordinates></Point>")
            partes.append("</Placemark>")
    partes.append("</Document></kml>")
    out_path = f"{nombre}_tracks.kml"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(partes))
    print(f"KML básico guardado: {out_path}")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    # 1) Asegurar capas NE (para provincias/ciudades/ríos/rutas)
    verificar_y_descargar_capas(shapefiles, carpeta_capas="Capas")

    # 2) Preparar mapa Basemap (también alimenta is_land)
    extent = _normalize_extent(bounding_box)
    m = _crear_mapa(extent)

    # 3) Generar por punto: PNG + CSV + KML (nombre único)
    for nombre, info in puntos.items():
        lat, lon = info["coords"]
        coords = calcular_direcciones_por_punto_km_basemap(nombre, lat, lon, m, step_km=step_km_tracks)
        exportar_csv(nombre, coords)
        df = pd.DataFrame(coords)
        resultados = {nombre: df}
        out_png = f"mapa_golfo_{nombre}.png"
        graficar_mapa_basemap({nombre: info}, resultados, extent, out_png, m=m)
        exportar_kml(nombre, df)

    print("\nListo. Se generaron mapas y KML/CSV por punto.")
