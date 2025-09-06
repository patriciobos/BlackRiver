#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mapas TL (planta / transito) con Basemap, interpolación suave y capas NE:
- Interpolación RBF (thin-plate) + suavizado gaussiano (sin “asterisco radial”).
- Rutas, límites provinciales (admin_1 lines) y ríos (centerlines).
- Ciudades + punto especial por mapa (según archivo).
- Datapoints apagados por defecto.

Requiere: numpy, pandas, matplotlib, basemap, scipy, requests, pyshp
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, List
import re, io, zipfile

import numpy as np
import pandas as pd
import requests
import shapefile  # pyshp

import matplotlib
matplotlib.use("Agg")  # no abre ventana
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import Normalize

from scipy.interpolate import RBFInterpolator, griddata
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

import multiprocessing as mp
import glob

# =========================
# CONFIGURACIÓN (editar acá)
# =========================

# Archivos a procesar (100 Hz)
CSV_LIST = sorted(glob.glob("input-data/gsm-*.csv"))

# Carpeta de salida para los mapas
MAPAS_DIR = Path("mapas")
MAPAS_DIR.mkdir(parents=True, exist_ok=True)

# Columnas
LAT_COL = "lat"
LON_COL = "lon"
VAL_COL = "tl_zmin"

# Mostrar/ocultar capas
DIBUJAR_CAMPO_TL  = True     # interpolación
DIBUJAR_PUNTOS_TL = False    # datapoints medidos (apagado)
DIBUJAR_CIUDADES  = True
DIBUJAR_PUNTO_ESPECIAL = True

# Capas vectoriales Natural Earth (todas activadas por defecto)
DIBUJAR_LIMITES_PROVINCIALES = True
DIBUJAR_RUTAS = True
DIBUJAR_RIOS  = True

# Puntos de análisis (lat, lon)
puntos: Dict[str, Dict[str, object]] = {
    "p1_transito": {"coords": (-41.41972, -64.40194), "color": "red"},
    "p2_planta":   {"coords": (-41.11806, -65.09806), "color": "red"},
}

# Estilo
CMAP_NAME = "jet_r"
CMAP_MIN, CMAP_MAX = 50.0, 220.0
SEA_COLOR   = "#CFEFFF"
LAND_COLOR  = "#E6E6E6"
COAST_COLOR = "#222222"
COUNTRY_COLOR = "#444444"
GRID_COLOR  = "#777777"

DATA_MARKER = "^"
DATA_SIZE_PT = 10
DATA_SCATTER_S = DATA_SIZE_PT**2

PUNTO_MARKER = "^"
PUNTO_MS = 12
PUNTO_FACE_DEFAULT = "#E74C3C"
PUNTO_EDGE = "#000000"

# BBox fijo del Golfo San Matías (para el mapa)
USE_FIXED_BBOX = True
BBOX_LAT_MIN, BBOX_LAT_MAX = -43.0, -40.5
BBOX_LON_MIN, BBOX_LON_MAX = -66.0, -62.5

# BBox para la grilla de interpolación (puede ser más grande)
GRID_LAT_MIN, GRID_LAT_MAX = -44.0, -40.5  # Extiende solo la grilla hacia el sur
GRID_LON_MIN, GRID_LON_MAX = -66.0, -62.5

# Si no usás bbox fijo, se calcula a partir de los datos con este margen:
MARGEN_DEG = 0.35

# Grilla de interpolación
GRID_NX = 350
GRID_NY = 350

# Interpolación suave (RBF) + suavizado final
RBF_FUNCTION = "thin_plate_spline"   # suave y continuo
RBF_SMOOTH   = 100.0                 # regularización (↑ = mucho más suave)
RBF_NEIGHBORS = 100                  # más vecinos para suavizar
RBF_MAX_RADIUS_KM = 100.0            # descartar > radio desde el dato más cercano

# Suavizado gaussiano post-interpolación (en celdas de la grilla)
SUAVIZAR_GAUSS = True
GAUSS_SIGMA = 4.0                    # suavizado gaussiano fuerte

# Ciudades del GSM (etiqueta a la izquierda del marcador)
CUSTOM_PLACES: List[Tuple[str, float, float]] = [
    ("San Antonio Oeste", -40.732, -64.946),
    ("Las Grutas",        -40.803, -65.083),
    ("Bahía Creek",       -41.0836, -63.9317),
    ("La Ensenada",       -41.155, -63.387),
    ("Playas Doradas",    -41.627, -65.024),
    ("Puerto Lobos",      -41.980, -65.050),
    ("Puerto Madryn",     -42.769, -65.038),
]

# Natural Earth (descarga auto a test/Capas)
NE_BASE = "https://naciscdn.org/naturalearth"
NATURAL_EARTH_LAYERS: Dict[str, Dict[str, str]] = {
    "coastline": {
        "url": f"{NE_BASE}/10m/physical/ne_10m_coastline.zip",
        "folder": "ne_10m_coastline",
        "shp": "ne_10m_coastline.shp",
    },
    "countries": {
        "url": f"{NE_BASE}/10m/cultural/ne_10m_admin_0_countries.zip",
        "folder": "ne_10m_admin_0_countries",
        "shp": "ne_10m_admin_0_countries.shp",
    },
    "admin_1_lines": {
        "url": f"{NE_BASE}/10m/cultural/ne_10m_admin_1_states_provinces_lines.zip",
        "folder": "ne_10m_admin_1_states_provincias_lines",
        "shp": "ne_10m_admin_1_states_provinces_lines.shp",
    },
    "roads": {
        "url": f"{NE_BASE}/10m/cultural/ne_10m_roads.zip",
        "folder": "ne_10m_roads",
        "shp": "ne_10m_roads.shp",
    },
    "rivers": {
        "url": f"{NE_BASE}/10m/physical/ne_10m_rivers_lake_centerlines.zip",
        "folder": "ne_10m_rivers_lake_centerlines",
        "shp": "ne_10m_rivers_lake_centerlines.shp",
    },
}
CAPAS_DIR = Path("test/Capas")

DPI = 300

# =========================
# HELPERS
# =========================

def ensure_layer(capas_dir: Path, layer_key: str) -> Optional[Path]:
    info = NATURAL_EARTH_LAYERS[layer_key]
    folder = capas_dir / info["folder"]
    shp_path = folder / info["shp"]
    if shp_path.exists():
        print(f"✔ Capa '{layer_key}' encontrada: {shp_path}")
        return shp_path
    url = info["url"]
    print(f"↓ Descargando capa '{layer_key}' desde {url}")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        zip_path = capas_dir / f"{info['folder']}.zip"
        # Si el zip ya existe, usarlo
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(folder)
        else:
            r = requests.get(url, timeout=60); r.raise_for_status()
            with open(zip_path, 'wb') as f:
                f.write(r.content)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(folder)
        if shp_path.exists():
            print(f"✔ Capa '{layer_key}' lista en: {shp_path}")
            return shp_path
        else:
            print(f"⚠ No se encontró el shapefile esperado tras extraer: {shp_path}")
    except Exception as e:
        print(f"⚠ Error descargando capa '{layer_key}': {e}")
    return None

def compute_bounds(lats: np.ndarray, lons: np.ndarray, margin_deg: float = 0.35) -> Tuple[float, float, float, float]:
    lat_min = float(np.nanmin(lats)) - margin_deg
    lat_max = float(np.nanmax(lats)) + margin_deg
    lon_min = float(np.nanmin(lons)) - margin_deg
    lon_max = float(np.nanmax(lons)) + margin_deg
    lat_min = max(-90, lat_min); lat_max = min(90, lat_max)
    lon_min = max(-180, lon_min); lon_max = min(180, lon_max)
    return lat_min, lat_max, lon_min, lon_max

def build_basemap(lat_min, lat_max, lon_min, lon_max) -> Basemap:
    return Basemap(projection="merc",
                   llcrnrlat=lat_min, urcrnrlat=lat_max,
                   llcrnrlon=lon_min, urcrnrlon=lon_max, resolution="i")

def draw_base(m: Basemap):
    m.drawmapboundary(fill_color=SEA_COLOR, zorder=0)
    m.fillcontinents(color=LAND_COLOR, lake_color=SEA_COLOR, zorder=3)  # tierra por encima del campo
    m.drawcoastlines(linewidth=0.7, color=COAST_COLOR, zorder=6)
    m.drawcountries(linewidth=0.5, color=COUNTRY_COLOR, zorder=6)

def draw_grid_top(m: Basemap):
    try:
        par = m.drawparallels(np.linspace(m.llcrnrlat, m.urcrnrlat, 5),
                              labels=[1,0,0,0], dashes=[2,2], fontsize=14,
                              color=GRID_COLOR, zorder=13)
        mer = m.drawmeridians(np.linspace(m.llcrnrlon, m.urcrnrlon, 5),
                              labels=[0,0,0,1], dashes=[2,2], fontsize=14,
                              color=GRID_COLOR, zorder=13)
        for d in (par, mer):
            for _, artists in d.items():
                for a in artists:
                    try: a.set_zorder(13)
                    except Exception: pass
    except Exception:
        pass

def annotate_custom_places(m: Basemap):
    for name, lat, lon in CUSTOM_PLACES:
        x, y = m(lon, lat)
        plt.plot([x],[y], marker="o", ms=10, mfc="#222222", mec="#FFFFFF", mew=0.6, zorder=12)
        plt.text(x-6000, y, name, fontsize=8, color="#111",
                 ha="right", va="center", zorder=12)

# ---- dibujo robusto de shapefiles de líneas con pyshp (filtrado por bbox) ----

def _bbox_intersects(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> bool:
    axmin, aymin, axmax, aymax = a
    bxmin, bymin, bxmax, bymax = b
    return not (axmax < bxmin or bxmax < axmin or aymax < bymin or bymax < aymin)

def draw_shp_lines_pyshp(m: Basemap, shp_path: Path,
                         bbox: Tuple[float,float,float,float],
                         linewidth: float = 0.6, color: str = "#333333",
                         zorder: int = 7):
    """Dibuja shapefile de líneas usando pyshp; filtra por bbox lon/lat; robusto (sin GeoPandas)."""
    try:
        r = shapefile.Reader(str(shp_path))
    except Exception as e:
        print(f"⚠ No se pudo abrir {shp_path}: {e}")
        return

    lon_min, lon_max, lat_min, lat_max = bbox[2], bbox[3], bbox[0], bbox[1]
    bbox_ll = (lon_min, lat_min, lon_max, lat_max)

    for sh in r.shapes():
        if not sh.points:
            continue
        # bbox del shape: [xmin, ymin, xmax, ymax] en lon/lat
        sb = tuple(sh.bbox)  # (xmin, ymin, xmax, ymax)
        if not _bbox_intersects(sb, (bbox_ll[0], bbox_ll[1], bbox_ll[2], bbox_ll[3])):
            continue

        pts = sh.points
        parts = list(sh.parts) + [len(pts)]
        for i in range(len(parts)-1):
            seg = pts[parts[i]:parts[i+1]]
            if len(seg) < 2: continue
            lons = np.array([p[0] for p in seg], dtype=float)
            lats = np.array([p[1] for p in seg], dtype=float)
            x, y = m(lons, lats)
            good = np.isfinite(x) & np.isfinite(y)
            if good.sum() < 2: continue
            plt.plot(np.asarray(x)[good], np.asarray(y)[good],
                     linewidth=linewidth, color=color, zorder=zorder)

# =========================
# INTERPOLACIÓN
# =========================

def nearest_distance_km_grid(lons_p, lats_p, LONg, LATg) -> np.ndarray:
    """Distancia al punto de datos más cercano (km) usando cKDTree en coords lon/lat escaladas."""
    if len(lons_p) == 0:
        return np.full(LONg.shape, np.inf, dtype=float)
    lat0 = np.deg2rad(np.nanmean(lats_p))
    # Escalado equirectangular (aprox. local):
    px = np.cos(lat0) * np.deg2rad(lons_p)
    py = np.deg2rad(lats_p)
    gx = np.cos(lat0) * np.deg2rad(LONg.ravel())
    gy = np.deg2rad(LATg.ravel())
    tree = cKDTree(np.c_[px, py])
    d_rad, _ = tree.query(np.c_[gx, gy], k=1)
    d_km = d_rad * 6371.0088
    return d_km.reshape(LONg.shape)

def interpolar_suave_rbf(m: Basemap, lons, lats, vals):
    """RBF (TPS) con regularización + limitación por radio y suavizado gaussiano."""
    # Grilla lon/lat fija (bbox fijo o derivado)
    if USE_FIXED_BBOX:
        lon_grid = np.linspace(GRID_LON_MIN, GRID_LON_MAX, GRID_NX)
        lat_grid = np.linspace(GRID_LAT_MIN, GRID_LAT_MAX, GRID_NY)
    else:
        lat_min, lat_max, lon_min, lon_max = compute_bounds(lats, lons, MARGEN_DEG)
        lon_grid = np.linspace(lon_min, lon_max, GRID_NX)
        lat_grid = np.linspace(lat_min, lat_max, GRID_NY)

    LONg, LATg = np.meshgrid(lon_grid, lat_grid)

    # Datos válidos
    mask_valid = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(vals)
    X = np.c_[lons[mask_valid], lats[mask_valid]]
    y = vals[mask_valid].astype(float)
    if X.shape[0] < 3:
        Z = griddata(points=X, values=y, xi=(LONg, LATg), method="nearest")
    else:
        try:
            rbf = RBFInterpolator(
                X, y,
                kernel=RBF_FUNCTION,
                neighbors=min(RBF_NEIGHBORS, X.shape[0]),
                smoothing=RBF_SMOOTH
            )
            Z = rbf(np.c_[LONg.ravel(), LATg.ravel()]).reshape(LONg.shape)
        except Exception:
            Z = griddata(points=X, values=y, xi=(LONg, LATg), method="linear")

    # Limitar por distancia al dato más cercano
    dmin_km = nearest_distance_km_grid(lons[mask_valid], lats[mask_valid], LONg, LATg)
    Z = np.where(dmin_km <= RBF_MAX_RADIUS_KM, Z, np.nan)

    # Suavizado gaussiano (solo donde hay datos)
    if SUAVIZAR_GAUSS:
        mask = np.isfinite(Z)
        if np.any(mask):
            Zfill = Z.copy()
            med = np.nanmedian(Zfill[mask])
            Zfill[~mask] = med
            Zsmooth = gaussian_filter(Zfill, sigma=GAUSS_SIGMA)
            Z = np.where(mask, Zsmooth, np.nan)

    # Clip y máscara final
    Z = np.clip(Z, CMAP_MIN, CMAP_MAX, out=Z, where=np.isfinite(Z))
    Zm = np.ma.masked_invalid(Z)

    XI, YI = m(LONg, LATg)
    return XI, YI, Zm

def latlon_to_rphi(lat, lon, lat0, lon0):
    """Convierte lat/lon a coordenadas polares r (km), phi (rad) respecto a origen lat0/lon0."""
    R = 6371.0088  # radio tierra en km
    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)
    lat0_rad = np.deg2rad(lat0)
    # distancia equirectangular
    r = R * np.sqrt((dlat)**2 + (np.cos(lat0_rad)*dlon)**2)
    phi = np.arctan2(dlat, np.cos(lat0_rad)*dlon)
    return r, phi

def rphi_to_latlon(r, phi, lat0, lon0):
    """Convierte r (km), phi (rad) a lat/lon respecto a origen lat0/lon0."""
    R = 6371.0088
    lat0_rad = np.deg2rad(lat0)
    dlat = r * np.sin(phi) / R
    dlon = r * np.cos(phi) / (R * np.cos(lat0_rad))
    lat = lat0 + np.rad2deg(dlat)
    lon = lon0 + np.rad2deg(dlon)
    return lat, lon

def interpolar_suave_rphi(m, lons, lats, vals, lat0, lon0):
    """Interpolación RBF en coordenadas polares (r, phi) respecto a origen lat0/lon0. Depuración incluida."""
    # Convertir puntos a r, phi
    r, phi = latlon_to_rphi(lats, lons, lat0, lon0)
    print(f"Origen lat0, lon0: {lat0}, {lon0}")
    print(f"Primeros puntos r: {r[:5]}, phi: {phi[:5]}")
    # Grilla en r y phi
    r_grid = np.linspace(0, 100, GRID_NX)  # radio hasta 100 km
    phi_grid = np.linspace(-np.pi, np.pi, GRID_NY)
    Rg, Pg = np.meshgrid(r_grid, phi_grid)
    print(f"Grilla r: {r_grid[:5]} ... {r_grid[-5:]}")
    print(f"Grilla phi: {phi_grid[:5]} ... {phi_grid[-5:]}")
    # Interpolación
    mask_valid = np.isfinite(r) & np.isfinite(phi) & np.isfinite(vals)
    X = np.c_[r[mask_valid], phi[mask_valid]]
    y = vals[mask_valid].astype(float)
    print(f"Puntos válidos para interpolar: {X.shape[0]}")
    if X.shape[0] < 3:
        Z = griddata(points=X, values=y, xi=(Rg, Pg), method="nearest")
    else:
        try:
            rbf = RBFInterpolator(
                X, y,
                kernel=RBF_FUNCTION,
                neighbors=min(RBF_NEIGHBORS, X.shape[0]),
                smoothing=RBF_SMOOTH
            )
            Z = rbf(np.c_[Rg.ravel(), Pg.ravel()]).reshape(Rg.shape)
        except Exception as e:
            print(f"Error RBFInterpolator: {e}")
            Z = griddata(points=X, values=y, xi=(Rg, Pg), method="linear")
    # Suavizado gaussiano
    if SUAVIZAR_GAUSS:
        mask = np.isfinite(Z)
        if np.any(mask):
            Zfill = Z.copy()
            med = np.nanmedian(Zfill[mask])
            Zfill[~mask] = med
            Zsmooth = gaussian_filter(Zfill, sigma=GAUSS_SIGMA)
            Z = np.where(mask, Zsmooth, np.nan)
    # Clip y máscara final
    Z = np.clip(Z, CMAP_MIN, CMAP_MAX, out=Z, where=np.isfinite(Z))
    Zm = np.ma.masked_invalid(Z)
    # Convertir grilla r, phi a lat/lon
    lat_grid, lon_grid = rphi_to_latlon(Rg, Pg, lat0, lon0)
    print(f"lat_grid min/max: {np.nanmin(lat_grid)}, {np.nanmax(lat_grid)}")
    print(f"lon_grid min/max: {np.nanmin(lon_grid)}, {np.nanmax(lon_grid)}")
    XI, YI = m(lon_grid, lat_grid)
    print(f"XI min/max: {np.nanmin(XI)}, {np.nanmax(XI)}")
    print(f"YI min/max: {np.nanmin(YI)}, {np.nanmax(YI)}")
    return XI, YI, Zm

# =========================
# PUNTO ESPECIAL
# =========================

def extraer_tipo_archivo(nombre: str) -> str:
    s = nombre.lower()
    if "planta" in s: return "planta"
    if "transito" in s or "tránsito" in s: return "transito"
    return "otro"

def elegir_punto_especial(nombre_archivo: str) -> Optional[Tuple[float,float,str]]:
    """Devuelve (lat, lon, label) según nombre del archivo usando el dict 'puntos'."""
    tipo = extraer_tipo_archivo(nombre_archivo)
    if tipo == "planta" and "p2_planta" in puntos:
        lat, lon = puntos["p2_planta"]["coords"]
        return float(lat), float(lon), "p2_planta"
    if tipo == "transito" and "p1_transito" in puntos:
        lat, lon = puntos["p1_transito"]["coords"]
        return float(lat), float(lon), "p1_transito"
    return None

# =========================
# PLOTEO DE UN CSV
# =========================

def obtener_estacion_por_fecha(nombre_archivo: str) -> str:
    """Devuelve la estación según el mes en el nombre del archivo."""
    m = re.search(r'_(\d{4})-(\d{2})-(\d{2})_', nombre_archivo)
    if m:
        mes = int(m.group(2))
        if mes == 2:
            return "verano"
        if mes == 8:
            return "invierno"
    return ""

def plot_one_csv(csv_path: Path):
    df = pd.read_csv(csv_path).dropna(subset=[LAT_COL, LON_COL, VAL_COL]).copy()
    if df.empty:
        print(f"✖ {csv_path.name}: sin datos válidos"); return

    lats = df[LAT_COL].to_numpy(float)
    lons = df[LON_COL].to_numpy(float)
    vals = df[VAL_COL].to_numpy(float)

    # BBox
    if USE_FIXED_BBOX:
        lat_min, lat_max = BBOX_LAT_MIN, BBOX_LAT_MAX
        lon_min, lon_max = BBOX_LON_MIN, BBOX_LON_MAX
    else:
        lat_min, lat_max, lon_min, lon_max = compute_bounds(lats, lons, MARGEN_DEG)

    # Figura y mapa
    plt.figure(figsize=(10, 9))
    m = build_basemap(lat_min, lat_max, lon_min, lon_max)
    draw_base(m)

    # Campo interpolado (debajo de líneas; tierra ya tapa costa)
    pcm = None
    if DIBUJAR_CAMPO_TL and len(vals) >= 3:
        XI, YI, Zm = interpolar_suave_rbf(m, lons, lats, vals)
        pcm = plt.pcolormesh(XI, YI, Zm, shading="gouraud",
                             cmap=CMAP_NAME, vmin=CMAP_MIN, vmax=CMAP_MAX, zorder=2)

    # Capas NE adicionales (líneas) con pyshp y filtro por bbox
    bbox_ll = (lat_min, lat_max, lon_min, lon_max)
    CAPAS_DIR.mkdir(parents=True, exist_ok=True)

    if DIBUJAR_LIMITES_PROVINCIALES:
        shp_admin = ensure_layer(CAPAS_DIR, "admin_1_lines")
        if shp_admin:
            draw_shp_lines_pyshp(m, shp_admin, bbox_ll, linewidth=0.7, color="#555555", zorder=8)

    if DIBUJAR_RUTAS:
        shp_roads = ensure_layer(CAPAS_DIR, "roads")
        if shp_roads:
            draw_shp_lines_pyshp(m, shp_roads, bbox_ll, linewidth=0.6, color="#8B4513", zorder=9)  # marrón

    if DIBUJAR_RIOS:
        shp_riv = ensure_layer(CAPAS_DIR, "rivers")
        if shp_riv:
            draw_shp_lines_pyshp(m, shp_riv, bbox_ll, linewidth=0.6, color="#1f77b4", zorder=9)

    # Ciudades (encima)
    if DIBUJAR_CIUDADES:
        for name, lat, lon in CUSTOM_PLACES:
            x, y = m(lon, lat)
            plt.plot([x],[y], marker="o", ms=10, mfc="#222222", mec="#FFFFFF", mew=0.6, zorder=12)
            if name in ["Bahía Creek", "La Ensenada"]:
                plt.text(x-6000, y+14000, name, fontsize=14, color="#111",
                         ha="left", va="top", zorder=12)
            else:
                plt.text(x-6000, y+6000, name, fontsize=14, color="#111",
                         ha="right", va="top", zorder=12)

    # Punto especial (encima)
    if DIBUJAR_PUNTO_ESPECIAL:
        pe = elegir_punto_especial(csv_path.name)
        if pe is not None:
            pe_lon, pe_lat, pe_label = pe
            color = PUNTO_FACE_DEFAULT
            # usar color del dict si existe
            if pe_label == "Planta":
                color = puntos["p2_planta"].get("color", color)
            elif pe_label == "Tránsito":
                color = puntos["p1_transito"].get("color", color)
            px, py = m(pe_lon, pe_lat)
            plt.plot(px, py, marker=PUNTO_MARKER, ms=PUNTO_MS,
                     mfc=color, mec=PUNTO_EDGE, mew=1.4, zorder=12)
            # etiqueta a la izquierda del marcador
            plt.text(px-6000, py, pe_label, fontsize=9, color="#111", ha="right", va="center", zorder=12)

    # Dibuja todos los puntos definidos en el vector 'puntos' sobre el mapa
    for nombre, info in puntos.items():
        lat, lon = info["coords"]
        color = info.get("color", PUNTO_FACE_DEFAULT)
        px, py = m(lon, lat)
        plt.plot(px, py, marker=PUNTO_MARKER, ms=PUNTO_MS,
                 mfc=color, mec=PUNTO_EDGE, mew=1.4, zorder=13)
        plt.text(px-6000, py, nombre, fontsize=14, color="#111", ha="right", va="center", zorder=13)

    # Puntos medidos (apagados por defecto)
    sc = None
    if DIBUJAR_PUNTOS_TL:
        x, y = m(lons, lats)
        sc = plt.scatter(x, y, c=np.clip(vals, CMAP_MIN, CMAP_MAX),
                         s=DATA_SCATTER_S, cmap=CMAP_NAME,
                         vmin=CMAP_MIN, vmax=CMAP_MAX,
                         marker=DATA_MARKER, edgecolors="none", zorder=11)

    # Grilla arriba de todo
    draw_grid_top(m)

    # Colorbar vertical con título horizontal
    mappable = pcm if (pcm is not None) else sc
    if mappable is not None:
        cbar = plt.colorbar(mappable, orientation="vertical", pad=0.02, shrink=0.85)
        cbar.mappable.set_clim(CMAP_MIN, CMAP_MAX)
        cbar.update_normal(cbar.mappable)
        cbar.ax.set_xlabel("TL(dB)", labelpad=8, loc="right")
        label = cbar.ax.xaxis.get_label()
        label.set_x(4.0)  # Ajusta el valor para mover el texto más a la derecha

    # Título
    fname = csv_path.name
    fdate = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
    fstr  = re.search(r"f=(\d{1,4}(?:[.,]\d+)?)\s*Hz", fname, re.IGNORECASE)
    date_str = fdate.group(1) if fdate else ""
    freq_str = (fstr.group(1).replace(",", ".")+" Hz") if fstr else "100 Hz"
    estacion = obtener_estacion_por_fecha(csv_path.name)
    plt.title(f"Pérdidas por transmisión — TL_min, f = {freq_str}, estación: {estacion}", pad=30)

    # Etiquetas de coordenadas lat/lon más grandes
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=22)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.10)

    # Guardar
    out_path = MAPAS_DIR / (csv_path.stem + "_map.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"✔ Figura guardada en {out_path}")

# =========================
# MAIN
# =========================

def process_csv(csv):
    plot_one_csv(Path(csv))

def main():
    plt.rcParams.update({'font.size': 16})
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(process_csv, CSV_LIST)

if __name__ == "__main__":
    main()
