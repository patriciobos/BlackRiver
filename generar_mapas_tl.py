#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generación de mapas de Pérdidas por Transmisión (TL) en el Golfo San Matías.

- Estructura de carpetas (rutas RELATIVAS a este script):
    low-csv/csv/{punto}/{fecha}/
    mid-csv/csv/{punto}/{fecha}/
  con archivos: gsm-planta_2023-02-15_f=10 Hz.csv   (un CSV por frecuencia)

- TL: elegir columna (p.ej. tl_5m, tl_10m, tl_20m, ...)

- Interpolación: 'linear' (griddata) o 'kriging' (PyKrige).

- Máscara: interpolar SOLO en mar, usando Capas:
    1) Si hay *ocean* en ./Capas -> usar esos polígonos.
    2) Si no, usar *land* y se invierte (mar = no tierra).
    3) Fallback: Natural Earth (tierra).
  La máscara se calcula por-proceso (cacheada), no se pasa por pickling.

- Dibujo con Cartopy (fondos Natural Earth 110m): tierra, océano, costas, bordes.

- Exporta:
    * PNG por mapa
    * GeoTIFF (EPSG:4326) de la grilla interpolada

Requisitos clave: numpy, pandas, scipy, matplotlib, cartopy, geopandas, rasterio, shapely, pykrige (opcional)
"""

# ===== Backend no interactivo para estabilidad (también en procesos hijos) =====
import matplotlib
matplotlib.use("Agg")

import os
import re
import math
import warnings
from pathlib import Path
from functools import lru_cache
import multiprocessing as mp

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Point
from shapely.strtree import STRtree

from scipy.interpolate import griddata

# GeoTIFF
import rasterio
from rasterio.transform import from_origin

# Kriging (opcional)
try:
    from pykrige.ok import OrdinaryKriging
    HAS_KRIGE = True
except Exception:
    HAS_KRIGE = False

# Cartopy
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Si descargaste Natural Earth en ./Capas, Cartopy lo va a tomar de ahí:
cartopy.config['data_dir'] = str(Path("./Capas").resolve())

# ==========================
# ===== PARÁMETROS =========
# ==========================

# Selección de corrida
PROCESS_MODE = "all"   # "all" | "filter"
FILTER_PUNTO = None    # "planta" | "transito" | None
FILTER_FECHA = None    # "2023-02-15" | "2023-08-15" | None
FILTER_RANGO = None    # "low" | "mid" | None
FILTER_FREQ_HZ = None  # p.ej. 10 | 270 | None

# Columna de TL a usar (tal cual aparece en los CSV)
TL_COLUMN = "tl_20m"

# Bounding box (lon_min, lon_max, lat_min, lat_max) en WGS84
# Poné None para automático a partir de los datos
bounding_box = [-65.5, -63.5, -42.5, -40.5]  # normalizado (min<max). Cambia si querés.

# Puntos a dibujar (lon, lat)
puntos = {
    "p1_transito": (-64.40194, -41.41972),
    "p2_planta"  : (-65.09806, -41.11806),
}
PUNTO_COLOR = "red"

# Visualización TL (dB)
VFIXED = True          # True usa VMIN/VMAX fijos
VMIN = 50.0
VMAX = 110.0
CMAP = "jet_r"         # jet invertida

# Interpolación
METHOD = "linear"      # "linear" | "kriging"
KRIGE_VARIOGRAM = "spherical"
KRIGE_NLAGS = 12

# Grilla
AUTO_GRID = True
GRID_STEP_DEG = 0.01    # si AUTO_GRID=False
GRID_MIN_STEP = 0.002
GRID_MAX_STEP = 0.05

# Máscara de mar
MASK_MODE = "ocean"   # "ocean" | "land" | None
MASK_LAND = bool(MASK_MODE)
CAPAS_DIR = Path("./Capas")

# Dibujo Cartopy (escala 110m para performance)
CARTOPY_SCALE = "110m"

# Exportes
OUT_DIR = Path("./out")
SAVE_GEO_TIFF = True
DPI = 160

# Multiproceso (estable con spawn)
N_WORKERS = max(1, mp.cpu_count())

# ==========================
# ===== ESTRUCTURA FS ======
# ==========================

RANGOS = {"low": (10, 250), "mid": (270, 570)}

CSV_ROOTS = {
    "low": Path("./low-csv/csv"),
    "mid": Path("./mid-csv/csv"),
}

CSV_PATTERN = re.compile(
    r"gsm-(?P<punto>planta|transito)_(?P<fecha>\d{4}-\d{2}-\d{2})_f=(?P<freq>\d+)\s*Hz\.csv$",
    re.IGNORECASE
)

# ==========================
# ====== UTILIDADES ========
# ==========================

def normalize_bbox(bbox):
    if bbox is None:
        return None
    lon_min, lon_max, lat_min, lat_max = bbox
    return [min(lon_min, lon_max), max(lon_min, lon_max),
            min(lat_min, lat_max), max(lat_min, lat_max)]

def find_csv_tasks():
    tasks = []
    for rango, root in CSV_ROOTS.items():
        if FILTER_RANGO and rango != FILTER_RANGO:
            continue
        for punto in ("planta", "transito"):
            if FILTER_PUNTO and punto != FILTER_PUNTO:
                continue
            base = root / punto
            if not base.exists():
                continue
            for fecha_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
                fecha = fecha_dir.name
                if FILTER_FECHA and fecha != FILTER_FECHA:
                    continue
                for fcsv in sorted(fecha_dir.glob("*.csv")):
                    m = CSV_PATTERN.search(fcsv.name)
                    if not m:
                        continue
                    freq = int(m.group("freq"))
                    if FILTER_FREQ_HZ is not None and freq != FILTER_FREQ_HZ:
                        continue
                    tasks.append((rango, punto, fecha, freq, fcsv))
    return tasks

def auto_grid_step(lons, lats):
    if len(lons) < 5:
        return GRID_STEP_DEG
    coords = np.column_stack([lons, lats])
    dmins = []
    for i in range(len(coords)):
        di = np.sqrt(((coords[i] - coords)**2).sum(axis=1))
        di = di[di > 0]
        if len(di):
            dmins.append(di.min())
    if not dmins:
        return GRID_STEP_DEG
    med = float(np.median(dmins))
    step = med / 2.0
    return min(max(step, GRID_MIN_STEP), GRID_MAX_STEP)

# ---------- Máscara de mar ----------
def choose_mask_file():
    """
    Devuelve (path_shp, mode) donde:
      - path_shp: str o None
      - mode: 'ocean' | 'land' | None
    No carga geometrías aquí; solo elige.
    """
    if not MASK_LAND or MASK_MODE is None:
        return None, None

    def find_shp(keyword):
        if not CAPAS_DIR.exists():
            return None
        bad = ("coastline", "boundary_lines", "roads", "rivers", "lake", "places", "admin")
        cands = []
        for shp in CAPAS_DIR.glob("**/*.shp"):
            name = shp.name.lower()
            if keyword in name and not any(b in name for b in bad):
                cands.append(shp)
        return str(sorted(cands)[0]) if cands else None

    if MASK_MODE == "ocean":
        shp = find_shp("ocean")
        if shp:  return shp, "ocean"
        shp = find_shp("land")
        if shp:  return shp, "land"
    else:  # "land"
        shp = find_shp("land")
        if shp:  return shp, "land"
        shp = find_shp("ocean")
        if shp:  return shp, "ocean"

    # Fallback: naturalearth_lowres (tierra)
    try:
        ne_path = gpd.datasets.get_path("naturalearth_lowres")
        return ne_path, "land"
    except Exception:
        return None, None

@lru_cache(maxsize=2)
def load_geoms_cached(path_shp):
    """Carga polígonos a EPSG:4326, dissolve+explode; cacheado por proceso."""
    if path_shp is None:
        return None
    gdf = gpd.read_file(path_shp)
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        return None
    gdf = gdf.dissolve().explode(index_parts=False, ignore_index=True)
    return gdf.geometry

def raster_mask_from_polys(xs, ys, geoms, mode):
    """
    Máscara booleana para la grilla (True = interpolar).
    - mode 'ocean': True dentro de polígonos de océano.
    - mode 'land' : True fuera de polígonos de tierra.
    Soporta STRtree.query que devuelve geometrías o índices.
    """
    if geoms is None or len(geoms) == 0 or mode is None:
        return np.ones((len(ys), len(xs)), dtype=bool)

    # convertir a lista indexable
    try:
        list_geoms = list(geoms)
    except Exception:
        list_geoms = list(getattr(geoms, "geometry", geoms))

    tree = STRtree(list_geoms)
    mask = np.zeros((len(ys), len(xs)), dtype=bool)

    def _resolve_candidates(cands):
        import numpy as _np
        if hasattr(cands, "dtype") and _np.issubdtype(cands.dtype, _np.integer):
            return [list_geoms[int(i)] for i in cands.tolist()]
        return list(cands)

    pred = "contains" if mode == "ocean" else None

    for j, lat in enumerate(ys):
        for i, lon in enumerate(xs):
            pt = Point(float(lon), float(lat))
            try:
                candidates = tree.query(pt, predicate=pred) if pred else tree.query(pt)
            except TypeError:
                candidates = tree.query(pt)
            geoms_cand = _resolve_candidates(candidates)

            if mode == "ocean":
                inside = any(poly.contains(pt) for poly in geoms_cand)
                mask[j, i] = inside
            else:  # land
                on_land = any(poly.contains(pt) for poly in geoms_cand)
                mask[j, i] = not on_land

    return mask

# ---------- Interpolación ----------
def interp_linear(lons, lats, vals, xs, ys):
    grid_x, grid_y = np.meshgrid(xs, ys)
    zi = griddata(points=np.column_stack([lons, lats]),
                  values=vals,
                  xi=(grid_x, grid_y),
                  method="linear")
    return zi

def interp_kriging(lons, lats, vals, xs, ys):
    if not HAS_KRIGE:
        warnings.warn("PyKrige no disponible. Usando 'linear'.")
        return interp_linear(lons, lats, vals, xs, ys)
    OK = OrdinaryKriging(
        lons, lats, vals,
        variogram_model=KRIGE_VARIOGRAM,
        nlags=KRIGE_NLAGS,
        verbose=False, enable_plotting=False
    )
    z, _ = OK.execute("grid", xs, ys)
    return np.asarray(z)

# ---------- GeoTIFF ----------
def save_geotiff(path_tif, xs, ys, z):
    xs_sorted = np.sort(xs)
    ys_sorted = np.sort(ys)
    px = float(np.mean(np.diff(xs_sorted)))
    py = float(np.mean(np.diff(ys_sorted)))
    transform = from_origin(xs_sorted.min(), ys_sorted.max(), px, py)

    profile = {
        "driver": "GTiff",
        "height": z.shape[0],
        "width": z.shape[1],
        "count": 1,
        "dtype": rasterio.float32,
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "lzw",
        "nodata": np.float32(np.nan),
    }
    with rasterio.open(path_tif, "w", **profile) as dst:
        dst.write(z.astype(np.float32), 1)

# ---------- Dibujo con Cartopy ----------
def plot_map(xs, ys, z, bbox, title, puntos_dict, out_png):
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(8.8, 7.6), dpi=DPI)
    ax = plt.axes(projection=proj)
    ax.set_extent([bbox[0], bbox[1], bbox[2], bbox[3]], crs=proj)

    # Fondo
    ax.add_feature(cfeature.OCEAN.with_scale(CARTOPY_SCALE), zorder=0, facecolor="#cfe7ff")
    ax.add_feature(cfeature.LAND.with_scale(CARTOPY_SCALE), zorder=0, facecolor="#e6e6e6")
    ax.coastlines(resolution=CARTOPY_SCALE, linewidth=0.8, zorder=10)
    # ax.add_feature(cfeature.BORDERS.with_scale(CARTOPY_SCALE), linewidth=0.5, edgecolor="#666", zorder=11)

    # Raster TL
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    im = ax.imshow(
        z, origin="lower", extent=extent,
        cmap=CMAP, vmin=VMIN if VFIXED else None, vmax=VMAX if VFIXED else None,
        interpolation="bilinear", transform=proj, zorder=5, aspect="auto"
    )

    # Puntos
    for name, (lon, lat) in puntos_dict.items():
        ax.plot(lon, lat, marker="o", ms=6, color=PUNTO_COLOR, transform=proj, zorder=20)
        ax.text(lon, lat, f" {name}", fontsize=8, color=PUNTO_COLOR, transform=proj, zorder=21)

    # Barra y grilla
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("TL [dB]")
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.4, linestyle="--")
    gl.top_labels = False; gl.right_labels = False

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# ==========================
# ========== MP ============
# ==========================
G_MASK_PATH = None
G_MASK_MODE = None

def _child_init(mask_path, mask_mode):
    # Reforzar backend en hijos
    import matplotlib
    matplotlib.use("Agg")
    global G_MASK_PATH, G_MASK_MODE
    G_MASK_PATH = mask_path
    G_MASK_MODE = mask_mode

# ==========================
# ======== WORKER ==========
# ==========================
def process_one(task):
    rango, punto, fecha, freq, path_csv = task
    df = pd.read_csv(path_csv)

    # Detectar lon/lat
    lon_col = next((c for c in ("lon","longitude","x","lon_wgs84") if c in df.columns), None)
    lat_col = next((c for c in ("lat","latitude","y","lat_wgs84") if c in df.columns), None)
    if lon_col is None or lat_col is None:
        raise ValueError(f"No encuentro columnas lon/lat en {path_csv.name}")
    if TL_COLUMN not in df.columns:
        raise ValueError(f"TL '{TL_COLUMN}' no está en {path_csv.name}. Disponibles: {', '.join([c for c in df.columns if c.lower().startswith('tl')])}")

    lons = df[lon_col].to_numpy(dtype=float)
    lats = df[lat_col].to_numpy(dtype=float)
    vals = df[TL_COLUMN].to_numpy(dtype=float)

    # BBox
    bbox = normalize_bbox(bounding_box)
    if bbox is None:
        lon_min, lon_max = np.nanmin(lons), np.nanmax(lons)
        lat_min, lat_max = np.nanmin(lats), np.nanmax(lats)
        pad_lon = (lon_max - lon_min) * 0.05
        pad_lat = (lat_max - lat_min) * 0.05
        bbox = [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat]

    # Grilla
    step = auto_grid_step(lons, lats) if AUTO_GRID else GRID_STEP_DEG
    xs = np.arange(bbox[0], bbox[1] + step*0.5, step)
    ys = np.arange(bbox[2], bbox[3] + step*0.5, step)

    # Interpolación
    if METHOD == "kriging":
        zi = interp_kriging(lons, lats, vals, xs, ys)
    else:
        zi = interp_linear(lons, lats, vals, xs, ys)

    # Máscara
    if MASK_LAND:
        geoms = load_geoms_cached(G_MASK_PATH)
        mask = raster_mask_from_polys(xs, ys, geoms, G_MASK_MODE)
        zi = np.where(mask, zi, np.nan)

    # Salidas
    out_dir = OUT_DIR / rango / punto / fecha
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"gsm-{punto}_{fecha}_f={freq}Hz"
    out_png = out_dir / f"{base}.png"
    out_tif = out_dir / f"{base}.tif"

    # Plot
    title = f"TL [{TL_COLUMN}] - {punto} {fecha} - {rango} f={freq} Hz"
    plot_map(xs, ys, zi, bbox, title, puntos, out_png)

    # GeoTIFF
    if SAVE_GEO_TIFF:
        save_geotiff(out_tif, xs, ys, zi)

    return str(out_png), (str(out_tif) if SAVE_GEO_TIFF else None)

# ==========================
# ========== MAIN ==========
# ==========================
def main():
    # Validaciones
    if METHOD not in ("linear", "kriging"):
        raise ValueError("METHOD debe ser 'linear' o 'kriging'")
    if METHOD == "kriging" and not HAS_KRIGE:
        warnings.warn("Seleccionaste 'kriging' pero PyKrige no está disponible. Se usará 'linear'.")

    tasks = find_csv_tasks()
    if not tasks:
        print("No se encontraron CSVs que cumplan los filtros/rutas.")
        return

    # Elegir shapefile para máscara (ruta + modo)
    mask_path, mask_mode = choose_mask_file()

    print(f"Tareas: {len(tasks)} | Workers: {N_WORKERS}")
    results = []

    if N_WORKERS > 1 and len(tasks) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=N_WORKERS,
                      initializer=_child_init,
                      initargs=(mask_path, mask_mode)) as pool:
            for r in pool.imap_unordered(process_one, tasks):
                results.append(r)
    else:
        _child_init(mask_path, mask_mode)  # inicializa globals también en corrida 1 hilo
        for t in tasks:
            results.append(process_one(t))

    print("\nArchivos generados:")
    for png, tif in results:
        if tif:
            print(f" - {png}\n   {tif}")
        else:
            print(f" - {png}")

if __name__ == "__main__":
    main()
