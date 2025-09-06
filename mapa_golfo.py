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
from cartopy.feature import GSHHSFeature
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

# Archivos shapefile locales (rutas relativas dentro de Capas/)
shapefiles = {
    "ne_50m_coastline": "Capas/ne_50m_coastline",
    "ne_50m_land": "Capas/ne_50m_land",
    "ne_50m_ocean": "Capas/ne_50m_ocean",
    "ne_50m_admin_0_boundary_lines_land": "Capas/ne_50m_admin_0_boundary_lines_land",
    "ne_10m_populated_places": "Capas/ne_10m_populated_places",
    "ne_10m_rivers_lake_centerlines": "Capas/ne_10m_rivers_lake_centerlines",
    "ne_10m_roads": "Capas/ne_10m_roads",
    # OJO: usamos "provincias" porque tu dict original lo usa así;
    # el downloader ya aliasa a "provinces".
    "ne_10m_admin_1_states_provincias": "Capas/ne_10m_admin_1_states_provincias",
}

# Cantidad de puntos intermedios entre el origen y el punto extremo (por dirección)
N_INTERMEDIOS = 10
# Incluir también el origen y el extremo en el CSV
INCLUIR_ORIGEN_Y_EXTREMO = True

# Bounding box CORRECTO: [lon_min, lon_max, lat_min, lat_max]
bounding_box = [-65.5, -63.5, -42.5, -40.5]

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
# HELPERS
# -------------------------------
def _normalize_extent(ext):
    """Devuelve [lon_min, lon_max, lat_min, lat_max] con min<=max en ambos ejes."""
    lon0, lon1, lat0, lat1 = ext
    return [min(lon0, lon1), max(lon0, lon1), min(lat0, lat1), max(lat0, lat1)]

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

def _geom_within_extent(geom, extent):
    """Devuelve True si la geometría toca el bbox [lon_min, lon_max, lat_min, lat_max]."""
    lon_min, lon_max, lat_min, lat_max = _normalize_extent(extent)
    bbox = sgeom.box(lon_min, lat_min, lon_max, lat_max)
    return geom.intersects(bbox)

# -------------------------------
# CARGA DE CAPAS
# -------------------------------
def cargar_costa(path_zip_o_dir):
    """Carga geometría de costa desde shapefile local (.shp dentro de una carpeta)."""
    shp_path = resolve_shp(path_zip_o_dir)
    geoms = []
    with fiona.open(shp_path) as src:
        for feat in src:
            geoms.append(sgeom.shape(feat["geometry"]))
    return ops.unary_union(geoms)

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

# -------------------------------
# PLOTEOS
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
                                  linewidth=0.6, linestyle=":", zorder=2.5)
    except Exception as e:
        print(f"⚠️ Provincias no disponibles: {e}")

def plot_ciudades(ax, extent, min_pop=None):
    """
    Dibuja solo las ciudades argentinas que estén dentro del bounding box (extent).
    extent = [lon_min, lon_max, lat_min, lat_max]
    min_pop: si se especifica, filtra por población mínima.
    """
    ext = _normalize_extent(extent)
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
                    if ext[0] <= lon <= ext[1] and ext[2] <= lat <= ext[3]:
                        ax.plot(lon, lat, marker='.', color='k', ms=3, transform=ccrs.PlateCarree(), zorder=4)
                        ax.text(lon+0.05, lat+0.05, str(name), fontsize=8, transform=ccrs.PlateCarree(), zorder=4.1)
    except Exception as e:
        print(f"⚠️ No pude cargar ciudades desde {shapefiles['ne_10m_populated_places']}: {e}")
        # Fallback manual
        for name, lat, lon in FALLBACK_CIUDADES:
            if ext[0] <= lon <= ext[1] and ext[2] <= lat <= ext[3]:
                ax.plot(lon, lat, marker='.', color='k', ms=3, transform=ccrs.PlateCarree(), zorder=4)
                ax.text(lon+0.05, lat+0.05, str(name), fontsize=8, transform=ccrs.PlateCarree(), zorder=4.1)

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
                if scalerank <= min_scalerank:
                    ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                                      facecolor="none", edgecolor="#2878B5",
                                      linewidth=1.0, zorder=3)
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
                                      facecolor="none", edgecolor="#D97706",
                                      linewidth=lw, zorder=3.2, linestyle="-")
    except Exception as e:
        print(f"⚠️ No pude cargar rutas: {e}")

# -------------------------------
# GEODÉSICAS Y CÁLCULOS
# -------------------------------
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
    pts = []
    if inter.geom_type == "Point":
        pts = [inter]
    elif inter.geom_type in ("MultiPoint", "GeometryCollection"):
        pts = [g for g in inter.geoms if g.geom_type == "Point"]
    elif inter.geom_type in ("LineString", "MultiLineString"):
        if inter.geom_type == "LineString":
            pts = [sgeom.Point(inter.coords[0])]
        else:
            pts = [sgeom.Point(list(inter.geoms)[0].coords[0])]
    if not pts:
        return None
    dmin = None
    pmin = None
    for p in pts:
        d = linea.project(p)
        if dmin is None or d < dmin:
            dmin = d; pmin = p
    return pmin

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
    az12, az21, dist_m = geod_local.inv(lon0, lat0, lon1, lat1)

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
        lonf, latf, _ = geod_local.fwd(lon0, lat0, az12, d)
        out.append((latf, lonf, f, d / 1000.0))  # (lat, lon, fracción, distancia_km)
    return out

# -------------------------------
# GRAFICADO
# -------------------------------
def dibujar_rosa_referencia(fig, anchor=(0.15, 0.15), size=0.18):
    axr = fig.add_axes([anchor[0], anchor[1], size, size])
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
    circ = mpatches.Circle((0,0), R, fill=False, lw=1)
    axr.add_patch(circ)
    axr.text(0, 0, "16-pt", ha='center', va='center', fontsize=8)

def graficar_mapa(puntos, resultados, costa, out_png=None):
    """Grafica el/los puntos y guarda el PNG en out_png (si no se da, arma uno con los nombres)."""
    # Determinar bounding box
    if bounding_box:
        extent = _normalize_extent(bounding_box)
    else:
        lats = [p["coords"][0] for p in puntos.values()]
        lons = [p["coords"][1] for p in puntos.values()]
        margin = 1.0
        extent = _normalize_extent([min(lons)-margin, max(lons)+margin,
                                    min(lats)-margin, max(lats)+margin])

    # Nombre por defecto si no vino uno
    if out_png is None:
        clave = "_".join(puntos.keys())
        out_png = f"mapa_golfo_{clave}.png"

    figsize = calcular_figsize(extent, base_height=8)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # 1) Fondo = mar
    ax.set_facecolor("lightblue")

    # 2) Tierra por encima (misma escala para evitar desajustes)
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "50m"),
        facecolor="lightgray", edgecolor="none", zorder=0.1
    )

    # 3) (Opcional) contorno de costa para “sellar” el borde
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor="black", zorder=1)

    # 4) Fronteras políticas
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linestyle="--", edgecolor="black", zorder=1.1)

    # opcional: lagos por encima de la tierra (mismo azul del fondo)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="lightblue", edgecolor="none", zorder=0.15)

    # Alternativa: costa GSHHS (más detallada)
    ax.add_feature(GSHHSFeature(scale="intermediate"), linewidth=0.5, edgecolor="black", facecolor="none", zorder=1)

    gl = ax.gridlines(draw_labels=True, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    # Overlays
    plot_rios(ax, extent, min_scalerank=5)
    plot_rutas(ax, extent, allowed_types=("Major Highway","Road"))
    plot_ciudades(ax, extent, min_pop=20000)
    plot_provincias(ax, extent)

    ax.text(-66, -41, "ARGENTINA", fontsize=14, weight="bold", color="gray",
            transform=ccrs.PlateCarree(),
            path_effects=[patheffects.withStroke(linewidth=3, foreground="white")], zorder=2)

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
                    s=30, facecolor="yellow", edgecolor="black",
                    transform=ccrs.PlateCarree(), zorder=3)
            if not sub.empty:
                lat_end, lon_end = sub.iloc[-1][["lat", "lon"]]
                ax.plot(lon_end, lat_end, marker="o", markersize=8,
                        markeredgecolor="red", markerfacecolor="white",
                        transform=ccrs.PlateCarree(), zorder=4)

    # Inset Atlántico Sur
    sub_ax = fig.add_axes((0.70, 0.70, 0.25, 0.25), projection=ccrs.PlateCarree())
    atlantico_sur_extent = _normalize_extent([-70, -40, -60, -30])
    sub_ax.set_extent(atlantico_sur_extent, crs=ccrs.PlateCarree())
    sub_ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="lightblue", zorder=0)
    sub_ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="lightgray", zorder=0.1)
    sub_ax.coastlines(zorder=1)
    extent_inset = ax.get_extent()
    rect = sgeom.box(extent_inset[0], extent_inset[2], extent_inset[1], extent_inset[3])
    sub_ax.add_geometries([rect], crs=ccrs.PlateCarree(), facecolor="none", edgecolor="red", zorder=20)

    dibujar_rosa_referencia(fig)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)  # importante para no acumular figuras en el loop
    print(f"Mapa guardado en: {out_png}")

    # Nota: si querés los “marcadores adicionales” que tenías al final,
    # podés mantener ese bloque ANTES de guardar el PNG.

def calcular_figsize(extent, base_height=8):
    """
    Ajusta el tamaño de la figura proporcional al bounding box.
    extent = [lon_min, lon_max, lat_min, lat_max]
    """
    lon_min, lon_max, lat_min, lat_max = _normalize_extent(extent)
    width = lon_max - lon_min
    height = lat_max - lat_min
    aspect = width / height if height != 0 else 1
    return (base_height * aspect, base_height)

# -------------------------------
# EXPORTACIÓN
# -------------------------------
def exportar_csv(nombre, coords):
    """Guarda las coordenadas en un CSV por punto, corrigiendo dist_km y asegurando formato decimal estándar."""
    df = pd.DataFrame(coords)
    if "dist_km" in df.columns:
        # Corrige valores sospechosos (>500 km) asumiendo que están en metros
        df["dist_km"] = df["dist_km"].apply(lambda x: x/1000 if x > 500 else x)
    df.to_csv(f"{nombre}_direcciones.csv", index=False, float_format='%.6f', encoding='utf-8', sep=',')

def exportar_kml(nombre, df):
    """
    Exporta KML por dirección. Usa simplekml si está instalado.
    Si no, genera un KML básico sin dependencias.
    """
    # Intento con simplekml si está disponible
    try:
        from simplekml import Kml, Style
        kml = Kml()
        style_line = Style(); style_line.linestyle.width = 2; style_line.linestyle.color = 'ff0000ff'  # rojo
        style_point = Style(); style_point.iconstyle.color = 'ff00ffff'  # amarillo/cian

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

    # ---- Fallback sin dependencias: genera un KML simple ----
    def _kml_escape(t):
        return (str(t)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

    line_color_abgr = "ff0000ff"  # rojo (ABGR en KML)
    point_color_abgr = "ff00ffff"  # amarillo/cian

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
    faltantes = []

    # 1) Asegurar capas descargadas / presentes
    verificar_y_descargar_capas(shapefiles, carpeta_capas="Capas")

    # 2) Cargar geometrías principales
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

    if costa is None:
        raise RuntimeError("Sin capa de costa no se puede continuar.")

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

    # 3) Generar mapas y KML por punto
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for nombre, info in puntos.items():
            lat, lon = info["coords"]
            coords = calcular_direcciones_por_punto_km(nombre, lat, lon, costa, executor=ex, step_km=step_km)
            exportar_csv(nombre, coords)
            df = pd.DataFrame(coords)
            resultados = {nombre: df}
            salida_png = f"mapa_golfo_{nombre}.png"
            graficar_mapa({nombre: info}, resultados, costa, out_png=salida_png)
            exportar_kml(nombre, df)

    # 4) Reporte final
    if faltantes:
        print("\n⚠️ Capas faltantes (descarga manual recomendada):")
        for capa in faltantes:
            print(f"  - {capa} (ver https://www.naturalearthdata.com)")
    else:
        print("\nTodas las capas principales fueron cargadas correctamente.")
