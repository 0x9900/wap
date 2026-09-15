#! /usr/bin/env python
# vim:fenc=utf-8
#
# Copyright © 2026 fred <github-fred@hidzz.com>
#
# Distributed under terms of the BSD 3-Clause license.

import argparse
import base64
import functools
import hashlib
import json
import logging
import os
import pickle
import re
import time
import urllib.request
import warnings
from datetime import UTC, datetime, timedelta
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from matplotlib import font_manager
from scipy.interpolate import pchip_interpolate, splev, splrep

__version__ = '0.1.0'

# Fix that!
try:
  from .webmap import plot_map
except ImportError:
  from webmap import plot_map

MAX_LOG_SIZE = 5_000_000

MAX_CACHE = 900
MIN_ROWS = 10
RESULT_DIR = Path('/var/tmp/results')
TRANS = ''.maketrans('/', '_')

SIGNATURE = '(C) {} Fred W6BSD - Antenna Analysis for {} - https://wspr.bsdworld.org/'

# These values are used by takeoff_angle()
R_OUTER = 1.0   # outer radius of the outermost ring
TRACK_W = 0.06  # width of each ring
GAP = 0.015  # gap between rings

WSPR_URL = 'https://db1.wspr.live/'
BAR_COLOR = 'tab:orange'

DARKBLUE = "#1d1330"
LIGHTBLUE = "#e1f7fa"
GRAY = "#4b4b4b"
LIGHTGRAY = "#ababab"

RC_PARAMS = {
  "axes.edgecolor": GRAY,
  "axes.facecolor": DARKBLUE,
  "axes.labelcolor": LIGHTGRAY,
  "axes.linewidth": 1,
  "figure.facecolor": DARKBLUE,
  "figure.figsize": (9, 5),
  "font.size": 10,
  "axes.grid": True,
  "grid.alpha": 0.7,
  "grid.linewidth": 0.25,
  "grid.linestyle": "dashed",
  "legend.fontsize": 6,
  "lines.linewidth": 1.5,
  "lines.markersize": 5,
  "text.color": "white",
  "xtick.color": LIGHTGRAY,
  "xtick.labelcolor": LIGHTGRAY,
  "xtick.minor.visible": False,
  "ytick.color": LIGHTGRAY,
  "ytick.labelcolor": LIGHTGRAY,
  "ytick.minor.visible": False,
}

# CMAP = plt.cm.Paired
# CMAP = plt.cm.tab20b
CMAP = plt.get_cmap('Accent')

plt.rcParams.update(RC_PARAMS)
BASE_DIR = Path(__file__).resolve().parent
FONT_ICELAND = BASE_DIR / 'static/Iceland-Regular.ttf'
FONT_JETBRAIN = BASE_DIR / 'static/JetBrainsMono-VariableFont_wght.ttf'
TITLE_FONT = font_manager.FontProperties(fname=FONT_ICELAND)
LEGEND_FONT = font_manager.FontProperties(fname=FONT_JETBRAIN)

FLIERPROPS = {
  "marker": 'o',
  "markerfacecolor": 'gray',
  "markeredgecolor": 'lightgray',
  "markersize": 3,
  "alpha": 0.7
}

HF_BANDS = set(["160m", "80m", "60m", "40m", "30m", "20m", "17m", "15m", "12m", "10m"])

_BANDS = [
  (-99, "All"),
  (1, "160m"),
  (3, "80m"),
  (5, "60m"),
  (7, "40m"),
  (10, "30m"),
  (14, "20m"),
  (18, "17m"),
  (21, "15m"),
  (24, "12m"),
  (28, "10m"),
  (50, "6m"),
  (70, "4m"),
  (144, "2m"),
  (432, "7cm"),
  (1296, "23cm"),
  (-1, "2200m"),
  (0, "640m"),
]


class BandLookup:
  def __init__(self, bands):
    self.bands = bands
    self.lookup_dict = {}
    for item in bands:
      if len(item) != 2:
        raise ValueError('The argument must be a tuple of 2 elements')
      _a, _b = item
      self.lookup_dict[_a] = _b
      self.lookup_dict[_b] = _a

  def __getitem__(self, key):
    if key not in self.lookup_dict:
      raise KeyError(f"{key} not found")
    return self.lookup_dict[key]

  def labels(self):
    return [v[1] for v in self.bands]


BANDS = BandLookup(_BANDS)


class BandValidator(argparse.Action):
  def __call__(self, parser, namespace, values, option_string=None):
    valid_bands = BANDS.labels()
    values = [v.lower() for v in values]

    if 'all' in values and len(values) > 1:
      parser.error("Cannot mix 'all' with specific bands")

    if 'all' in values:
      setattr(namespace, self.dest, ['all'])
    else:
      for band in values:
        if band not in valid_bands:
          parser.error(f"Band '{band}' not valid. Choose from {valid_bands}")
      setattr(namespace, self.dest, values)


def logger_setup(log_file, max_bytes=MAX_LOG_SIZE, count=5, level=None):
  format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  logger = logging.getLogger()
  level = level or os.getenv('LOG_LEVEL', 'INFO')
  logger.setLevel(level)
  logger.handlers.clear()

  handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=count)
  handler.setLevel(level)

  formatter = logging.Formatter(format_str)
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  # Capture scipy warnings
  logging.captureWarnings(True)
  return logger


def file_cache(cache_dir=".cache"):
  def decorator(func):
    # Create cache directory if it doesn't exist
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def _gen_key(func_name, args):
      temp = '-'.join(args)
      temp = re.sub(r'[^a-zA-Z0-9]', '-', temp)
      key_string = re.sub(r'-+', '-', temp)
      return f"wap-{func_name}-{key_string}"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      # Create a unique cache key from arguments
      cache_key = _gen_key(func.__name__, args)
      cache_file = Path(cache_dir) / f"{cache_key}.pkl"
      logging.info(cache_file)

      # Try to load from cache
      if cache_file.exists() and cache_file.stat().st_mtime + MAX_CACHE > time.time():
        try:
          with cache_file.open('rb') as f:
            result = pickle.load(f)
            logging.info("Cache hit: loaded from %s", cache_file)
            return result
        except Exception as err:  # pylint: disable=broad-exception-caught
          logging.info("Cache read error: %s", err)

      # Call the function and cache the result
      result = func(*args, **kwargs)

      try:
        with open(cache_file, 'wb') as f:
          pickle.dump(result, f)
        logging.info("Cache miss: saved to %s", cache_file)
      except Exception as err:  # pylint: disable=broad-exception-caught
        logging.info("Cache write error: %s", err)

      return result

    return wrapper
  return decorator


def log_calls(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    args_str = ', '.join(repr(arg) for arg in args)
    kwargs_str = ', '.join(f"{k}={v!r}" for k, v in kwargs.items())
    all_args = ', '.join(filter(None, [args_str, kwargs_str]))

    logging.info("Calling %s(%s)", func.__name__, all_args)
    result = func(*args, **kwargs)
    logging.info("%s returned %r", func.__name__, result)

    return result
  return wrapper


def wsprlive_get(call, start_date):
  now = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
  start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
  request = (
    f"SELECT * FROM wspr.rx WHERE tx_sign='{call}' "
    f"AND time >= '{start_date_str}' AND time < '{now}'"
  )
  logging.info(request)
  url = "https://db1.wspr.live/?query=" + urllib.parse.quote_plus(request + " FORMAT JSON")

  try:
    with urllib.request.urlopen(url) as response:
      if response.getcode() == 200:
        charset = response.headers.get_content_charset('utf-8')
        content = response.read()
        data = json.loads(content.decode(charset))['data']
        return pd.json_normalize(data)
  except urllib.error.URLError as err:
    logging.error("Failed to fetch data: %s", err)

  return pd.DataFrame([])


def get_start_date(delta):
  units = {'m': 'minutes',
           'h': 'hours',
           'd': 'days'}
  try:
    _delta, _unit = int(delta[:-1]), delta[-1].lower()
  except ValueError as err:
    logging.error(err)
    raise

  if _unit not in units:
    raise ValueError('The unit can only be (d, m, s)')

  now = datetime.now(UTC)
  _delta = timedelta(**{units[_unit]: _delta})
  if _delta.total_seconds() > 86400 * 20:
    raise ValueError('The delta time can only be 20 days max')

  start_time = (now - _delta).replace(second=0, microsecond=0)
  return start_time


@file_cache('/tmp')
def get_data(call, delta):
  start_time = get_start_date(delta)
  logging.info('loading data from wspr.live')
  data = wsprlive_get(call, start_time)
  if data.empty:
    return data

  cols = ['snr', 'distance', 'power', 'azimuth', 'frequency']
  data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')
  data = data.dropna(subset=cols)
  data = data[data["distance"] > 0.0]

  data["band"] = (
    (data["band"].astype(float))
    .map(dict(BANDS.bands))
  )
  data["time"] = pd.to_datetime(data["time"], format="%Y-%m-%d %H:%M:%S")
  return data


def format_band(value):
  return f"{int(value)}m" if value >= 1 else f"{value:.2f}m"


def set_title(ax, title):
  ax.set_title(title, fontproperties=TITLE_FONT, fontsize=22, color='#dfffdf')
  # ax.set_title(title, fontsize=20, color='#dfffdf')


def remove_outliers_iqr(df, value_col, group_col, k=1.2):
  df = df.copy()

  group_stats = df.groupby(group_col)[value_col]
  q1 = group_stats.transform(lambda x: x.quantile(0.25))
  q3 = group_stats.transform(lambda x: x.quantile(0.75))

  iqr = q3 - q1
  lower_bound = q1 - (k * iqr)
  upper_bound = q3 + (k * iqr)

  # Create a mask for values within the bounds
  mask = df[value_col].between(lower_bound, upper_bound)
  return df[mask]


def remove_global_outliers(df, value_col, k=3):
  q1 = df[value_col].quantile(0.25)
  q3 = df[value_col].quantile(0.75)
  iqr = q3 - q1
  mask = df[value_col].between(q1 - k*iqr, q3 + k*iqr)
  return df[mask]


def plot2string(call, fmt='webp'):
  year = datetime.now(UTC).year
  encode = {
    'png': "data:image/png;base64,",
    'svg': "data:image/svg+xml;base64,",
    'webp': "data:image/webp;base64,",
  }
  fig = plt.gcf()
  fig.text(.01, .015, SIGNATURE.format(year, call), fontsize=8, color='dimgray')
  buf = BytesIO()
  plt.savefig(buf, format=fmt, dpi=150)
  buf.seek(0)
  return encode[fmt] + base64.b64encode(buf.getvalue()).decode('ascii')


def smart_gap_handling(df_band, bin_width):
  """Intelligent gap detection combining multiple heuristics."""

  az_bins = df_band['az_bin'].unique()
  distances = df_band['distance'].values

  # Metric 1: Coverage
  coverage = len(az_bins) / (360 / bin_width)

  # Metric 2: Dynamic range
  dynamic_range = np.percentile(distances, 95) / np.percentile(distances, 5)

  # Metric 3: Actual gap distribution
  az_sorted = np.sort(az_bins)
  gaps = np.diff(np.append(az_sorted, az_sorted[0] + 360))
  max_gap = gaps.max()
  median_gap = np.median(gaps)

  # Decision logic
  if coverage > 0.8 and dynamic_range < 2:
    # Omni-like: small gaps are data issues
    gap_threshold = median_gap * 2
    fill_strategy = 'interpolate'  # Trust interpolation

  elif max_gap > 90 or coverage < 0.5:
    # Highly directional: large gaps are real
    gap_threshold = max(60, median_gap * 3)
    fill_strategy = 'anchor_low'  # Use low anchor points

  else:
    # Moderate directionality
    gap_threshold = median_gap * 2.5
    fill_strategy = 'anchor_adaptive'  # Use percentile

  return {
      'gap_threshold': gap_threshold,
      'fill_strategy': fill_strategy,
      'antenna_type': 'omni' if coverage > 0.8 else 'directional',
      'metrics': {
          'coverage': coverage,
          'dynamic_range': dynamic_range,
          'max_gap': max_gap
      }
  }


def radiation_pattern(data, call):
  """Generate polar radiation pattern plot with adaptive interpolation."""
  df = data[['band', 'distance', 'azimuth']].copy()
  df["azimuth"] = df["azimuth"] % 360
  df = remove_outliers_iqr(df, value_col="distance", group_col="band")
  df = df.sort_values('band', key=lambda x: x.str.rstrip('m').astype(int))

  _, ax = plt.subplots(figsize=(9, 7), subplot_kw={"projection": "polar"})

  bands = df["band"].unique()
  color_map = {b: CMAP(i / len(bands)) for i, b in enumerate(bands)}

  for band, sub in df.groupby("band", sort=False):
    _plot_band_pattern(ax, sub, band, color_map[band], df)

  ax.set_theta_zero_location("N")
  ax.set_theta_direction(-1)
  ax.set_ylim(bottom=0, top=df['distance'].max() * 1.1)
  ax.set_rlabel_position(315)
  ax.tick_params(axis="y", labelsize=8)
  ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), prop=LEGEND_FONT)
  set_title(ax, "Radiation Pattern")

  figure = plot2string(call)
  plt.close()
  return figure


def _plot_band_pattern(ax, sub, band, color, df_all):
  """Process and plot pattern for a single band."""
  bin_width = _calculate_bin_width(df_all)
  sub = _bin_and_aggregate(sub, bin_width)
  az, r = sub["az_bin"].to_numpy(), sub["distance"].to_numpy()

  if len(az) < 5:
    ax.plot(np.deg2rad(az), r, label=band, color=color)
    return

  gap_info = smart_gap_handling(sub, bin_width)
  az_final, r_final = _process_gaps(az, r, gap_info)
  theta_smooth, r_smooth = _interpolate_pattern(
    az_final, r_final, r.max(), gap_info, band
  )

  ax.plot(theta_smooth, r_smooth, linewidth=1, label=band, color=color)


def _calculate_bin_width(df):
  """Compute adaptive bin width based on data density."""
  median_spacing = df.groupby('band')['azimuth'].apply(
    lambda x: np.median(np.diff(np.sort(x.unique())))
  ).median()
  return max(5, min(15, median_spacing * 2))


def _bin_and_aggregate(sub, bin_width):
  """Bin azimuth data and aggregate distances."""
  sub["az_bin"] = (sub["azimuth"] // bin_width) * bin_width
  return sub.groupby(["band", "az_bin"], as_index=False).agg(
    distance=("distance", "max")
  ).sort_values("az_bin")


def _process_gaps(az, r, gap_info):
  """Add anchor points in coverage gaps based on antenna type."""
  gaps = np.diff(np.append(az, az[0] + 360))
  gap_threshold = gap_info['gap_threshold']

  if gap_info['fill_strategy'] == 'interpolate':
    return az, r

  az_list, r_list = az.tolist(), r.tolist()
  r_null = _calculate_anchor_value(r, gap_info['fill_strategy'])

  for i, gap in enumerate(gaps):
    if gap > gap_threshold:
      gap_center = (az[i] + gap / 2) % 360
      az_list.append(gap_center)
      r_list.append(r_null)

  az_final = np.array(az_list)
  r_final = np.array(r_list)
  idx = np.argsort(az_final)
  return az_final[idx], r_final[idx]


def _calculate_anchor_value(r, strategy):
  """Determine anchor point value for gaps."""
  if strategy == 'anchor_low':
    return max(np.percentile(r, 5), r.min() * 0.3)
  return np.percentile(r, 10)


def _interpolate_pattern(az_final, r_final, r_max, gap_info, band):
  """Create smooth interpolated pattern with overshoot handling."""
  theta_train = np.deg2rad(az_final)
  theta_smooth = np.linspace(0, 2 * np.pi, 720, endpoint=True)

  # Initial spline interpolation
  smoothing = len(az_final) * (3 if gap_info['antenna_type'] == 'omni' else 7)
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="The maximal number of iterations",
                            category=RuntimeWarning)
    tck = splrep(theta_train, r_final, s=smoothing, per=True)
  r_smooth = splev(theta_smooth, tck)

  # Handle overshoot
  if r_smooth.max() / r_max > 1.15:
    logging.warning("Band %s: Overshoot detected, using PCHIP interpolation", band)
    r_smooth = _fallback_pchip(theta_train, r_final, theta_smooth, r_max)

  return theta_smooth, r_smooth


def _fallback_pchip(theta_train, r_final, theta_smooth, r_max):
  """Use PCHIP interpolation for overshoot cases."""

  theta_extended = np.concatenate([
    theta_train - 2*np.pi, theta_train, theta_train + 2*np.pi
  ])
  r_extended = np.tile(r_final, 3)
  r_smooth = pchip_interpolate(theta_extended, r_extended, theta_smooth)

  return np.clip(r_smooth, 0, r_max * 1.05)


def azimuth_scatter(data, call, min_points_to_display=2):
  # pylint: disable=too-many-locals
  df = data[['band', 'azimuth', 'distance']].copy()
  df["azimuth"] = df["azimuth"] % 360

  bin_width = 2
  df["az_bin"] = (df["azimuth"] // bin_width) * bin_width

  # Calculate counts per (band, az_bin) group
  counts = (
    df.groupby(["band", "az_bin"])
    .size()
    .rename("count")
    .reset_index()
  )

  # Calculate average distance per group
  dist_p75 = (
    df.groupby(["band", "az_bin"])
    ["distance"]
    .quantile(0.75)
    .rename("dist_p75")  # Renamed for clarity
    .reset_index()
  )
  # Create aggregated dataframe
  aggregated = counts.merge(dist_p75, on=["band", "az_bin"], how="left")

  # Filter out bins with too few points
  aggregated = aggregated[aggregated["count"] >= min_points_to_display].copy()

  # Calculate representative azimuth (middle of the bin)
  aggregated["azimuth"] = aggregated["az_bin"] + bin_width / 2
  aggregated = aggregated.sort_values('band', key=lambda x: x.str.rstrip('m').astype(int))

  bands = aggregated["band"].unique()
  color_map = {b: CMAP(i / len(bands)) for i, b in enumerate(bands)}

  _, ax = plt.subplots(figsize=(9, 7), subplot_kw={"projection": "polar"})

  for band, sub in aggregated.groupby("band", sort=False):
    theta = np.deg2rad(sub["azimuth"])

    # Scale sizes based on count (log scale can help with large ranges)
    min_size, max_size = 10, 100
    # Use log scaling if count range is large
    counts = sub["count"]
    if counts.max() / counts.min() > 100:
      sizes = min_size + (max_size - min_size) * np.log10(counts) / np.log10(counts.max())
    else:
      sizes = min_size + (max_size - min_size) * counts / counts.max()

    # Option: Add small random jitter to reduce overplotting
    theta_jitter = theta + np.random.uniform(-0.01, 0.01, len(theta))
    distance_jitter = sub["dist_p75"] * np.random.uniform(0.99, 1.01, len(sub))

    ax.scatter(theta_jitter, distance_jitter, s=sizes, alpha=0.6,
               label=band, color=color_map[band], edgecolors='white', linewidth=0.5)

  ax.set_ylim(bottom=1)
  ax.set_theta_zero_location("N")
  ax.set_theta_direction(-1)
  ax.tick_params(axis="y", labelsize=8)
  ax.set_rlabel_position(310)
  ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), prop=LEGEND_FONT)
  ax.set_rscale("log")

  set_title(ax, "Azimuth distance and density (aggregated)")
  figure = plot2string(call)
  plt.close()
  return figure


def band_distance(data, call):
  band_dist = data[['band', 'distance']].copy()
  band_dist = remove_outliers_iqr(band_dist, value_col="distance", group_col="band", k=2.5)

  if band_dist.empty:
    logging.warning('No HF bands worked')
    return None

  ax = band_dist.boxplot(
    column='distance', by='band', patch_artist=True,
    boxprops={"facecolor": BAR_COLOR, "color": BAR_COLOR},
    medianprops={"color": 'w', "linewidth": 1.5},
    whiskerprops={"color": 'green'},
    capprops={"color": 'green'},
    flierprops=FLIERPROPS
  )

  plt.suptitle("")
  ax.set_xlabel(None)
  ax.set_ylabel("Distance (km)")
  ax.tick_params(axis='x', rotation=15)
  ax.ticklabel_format(style='plain', axis='y')

  set_title(ax, "Distance / Band")
  figure = plot2string(call)
  plt.close()
  return figure


def hours_distance(data, call):
  hour_dist = data[['time', 'distance']].copy()
  hour_dist['time'] = hour_dist.time.dt.hour
  hour_dist = remove_outliers_iqr(hour_dist, value_col="distance", group_col="time", k=2.5)

  if hour_dist.empty:
    logging.warning('No HF bands worked')
    return None

  ax = hour_dist.boxplot(
    column='distance', by='time', patch_artist=True,
    boxprops={"facecolor": BAR_COLOR, "color": BAR_COLOR},
    medianprops={"color": 'w', "linewidth": 1.5},
    whiskerprops={"color": 'green'},
    capprops={"color": 'green'},
    flierprops=FLIERPROPS
  )

  plt.suptitle("")
  ax.set_xlabel("UTC Hours")
  ax.set_ylabel('Distance / Hours')

  set_title(ax, "Distance / Hour")
  figure = plot2string(call)
  plt.close()
  return figure


def distance_to_angle(distance_km: float) -> float:
  """Convert distance (km) to approximate takeoff angle (degrees)."""
  breakpoints = [500, 1500, 2000, 5000, 7000, 20000]
  angles = [90,   30,   20,   10,    5,     0]
  return float(np.interp(distance_km, breakpoints, angles, left=90.0, right=10.0))


def takeoff_angle(data, call):
  data = data[['distance', 'band']]
  data['estimated_angle'] = data['distance'].apply(distance_to_angle)

  angles = data.groupby('band')['estimated_angle'].agg([
    lambda x: x.quantile(0.25),
    lambda x: x.quantile(0.75)
  ]).rename(columns={'<lambda_0>': 'q25', '<lambda_1>': 'q75'})
  angles = angles.reset_index()
  angles = angles.sort_values('band', key=lambda x: x.str.rstrip('m').astype(int))

  colors = {b: CMAP(i / angles.shape[0]) for i, b in enumerate(angles.band)}

  _, ax = plt.subplots()
  # --- Colored arcs for each band ---
  for i, row in enumerate(angles.itertuples()):
    r = R_OUTER - i * (TRACK_W + GAP)
    track = mpatches.Wedge((0, 0), r, theta1=0, theta2=90, width=TRACK_W,
                           color=LIGHTGRAY, alpha=0.05, zorder=1)
    arc = mpatches.Wedge((0, 0), r, theta1=row.q25, theta2=row.q75, width=TRACK_W,
                         color=colors[row.band], zorder=2, alpha=0.92)
    ax.add_patch(track)
    ax.add_patch(arc)

  for a in [0, 90]:
    rad = np.radians(a)
    ax.plot([0, 1.12 * np.cos(rad)], [0, 1.12 * np.sin(rad)], color=LIGHTGRAY,
            alpha=0.5, lw=1, ls="--", zorder=0)

  # --- Tick marks every 15° ---
  for a in range(0, 91, 15):
    rad = np.radians(a)
    if a not in (0, 90):
      ax.plot([0.3 * np.cos(rad), 1.08 * np.cos(rad)], [0.3 * np.sin(rad), 1.08 * np.sin(rad)],
              color=GRAY, lw=0.75, ls='--', zorder=0)
    ax.text(1.15 * np.cos(rad), 1.15 * np.sin(rad), f"{a}°", ha="center", va="center",
            fontsize=8, color=LIGHTGRAY)

  # --- Legend ---
  legend_patches = [
    mpatches.Patch(color=colors[row.band], label=f"{row.band}: {row.q25:.0f}° – {row.q75:.0f}°")
    for row in angles.itertuples()
  ]
  ax.legend(handles=legend_patches, bbox_to_anchor=(1.0, 0.9), prop=LEGEND_FONT)

  ax.set_xlim(-0.15, 1.68)
  ax.set_ylim(-0.15, 1.30)
  ax.set_aspect("equal")
  ax.axis("off")
  set_title(ax, "Antenna Takeoff Angle by Band")

  plt.tight_layout(pad=1)
  figure = plot2string(call)
  plt.close()
  return figure


def plot_snr_vs_distance(data, call):
  # Prepare data
  data = data[['snr', 'distance']].copy()
  data = data.sort_values('distance')

  # Create distance bins
  n_bins = 20
  data['distance_bin'] = pd.qcut(data['distance'], q=n_bins, duplicates='drop')

  # Calculate comprehensive statistics
  bin_stats = []
  for _, group in data.groupby('distance_bin'):
    if len(group) >= 3:
      bin_stats.append({
        'distance': group['distance'].median(),
        'snr_mean': group['snr'].mean(),
        'snr_median': group['snr'].median(),
        'snr_std': group['snr'].std(),
        'snr_max': group['snr'].max(),
        'snr_min': group['snr'].min(),
        'snr_10th': group['snr'].quantile(0.10),
        'snr_25th': group['snr'].quantile(0.25),
        'snr_75th': group['snr'].quantile(0.75),
        'snr_90th': group['snr'].quantile(0.90),
        'n_samples': len(group)
      })

  stats_df = pd.DataFrame(bin_stats)

  # Plot mean with std deviation band
  _, ax = plt.subplots()
  ax.plot(stats_df['distance'], stats_df['snr_mean'], 'tab:blue', label='Mean SNR')
  ax.fill_between(stats_df['distance'],
                  stats_df['snr_mean'] - stats_df['snr_std'],
                  stats_df['snr_mean'] + stats_df['snr_std'],
                  alpha=0.3, color='tab:red', label='Mean ± 1σ')

  # Plot IQR range (25th-75th percentile)
  ax.fill_between(stats_df['distance'], stats_df['snr_25th'], stats_df['snr_75th'],
                  alpha=0.2, color='tab:cyan', label='IQR (25th-75th)')

  # Plot median and percentiles
  ax.plot(stats_df['distance'], stats_df['snr_median'], color='tab:green', label='Median SNR')
  ax.plot(stats_df['distance'], stats_df['snr_10th'], color='tab:red',
          linestyle='dotted', label='10th Percentile')
  ax.plot(stats_df['distance'], stats_df['snr_90th'], color='tab:red',
          linestyle='dotted', label='90th Percentile')

  ax.set_xlabel('Distance (km)', fontsize=12)
  ax.set_ylabel('SNR (dB)', fontsize=12)
  set_title(ax, 'Signal Strength vs Distance')
  ax.legend(loc='best', fontsize=10, prop=LEGEND_FONT)

  plt.tight_layout(pad=1.8)
  figure = plot2string(call)
  plt.close()
  return figure


def snr_distance_errorbar(df, call):
  data = df.copy()
  max_distance = data["distance"].max()
  bins = [0, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000,
          8000, 12000, 16000, 22000, 30000]
  bins = [x for x in bins if x < max_distance] + [max_distance]

  data["distance_bin"] = pd.cut(data["distance"], bins=bins)

  stats = (
    data.groupby("distance_bin", observed=True)
    ["snr"]
    .agg(
      q25=lambda x: x.quantile(0.25),
      median="median",
      q75=lambda x: x.quantile(0.75),
      count="count"
    )
    .reset_index()
  )
  stats["distance_center"] = stats["distance_bin"].apply(lambda x: x.mid)

  _, ax = plt.subplots()

  ax.errorbar(
    stats["distance_center"], stats["median"],
    yerr=[
      stats["median"] - stats["q25"],
      stats["q75"] - stats["median"]
    ],
    color=BAR_COLOR, lw=1, fmt="o", capsize=3
  )
  ax2 = ax.twinx()

  ax.set_xlabel('Distance (km)', fontsize=12)
  ax.set_ylabel('SNR (dB)', fontsize=12)
  set_title(ax, 'SNR Distribution vs Distance')

  ax2.plot(stats["distance_center"], stats["count"], color=LIGHTBLUE, alpha=0.5)
  ax2.set_ylabel("Number of spots", fontsize=12)
  ax2.grid(True, color="green", alpha=0.9)

  plt.tight_layout(pad=1.8)
  figure = plot2string(call)
  plt.close()
  return figure


def draw_all(data, call):
  graph_functions = (
    (azimuth_scatter, 'Azimuth scatter'),
    (radiation_pattern, 'Aimuth distance'),
    (band_distance, 'Distance / Band'),
    (hours_distance, 'Distance / Hours'),
    (plot_snr_vs_distance, 'SNR vs distance'),
    (snr_distance_errorbar, 'SNR vs distance'),
    (takeoff_angle, 'Takeoff angle'),
  )
  graphs = {}
  for func, label in graph_functions:
    try:
      logging.info('Drawing: %s', label)
      if graph := func(data, call):
        graphs[func.__name__] = (label, graph)
      else:
        logging.warning('The call to %s returned None', func.__name__)
    except ValueError as err:
      logging.error('"%s": %s', label, err)
  return graphs


def render_html(data, output, call, antenna):
  template_dir = Path(__file__).with_name('templates')
  static_dir = Path(__file__).with_name('static')
  env = Environment(loader=FileSystemLoader([template_dir, static_dir]))

  template = env.get_template('perf.html')
  start_time = data.time.min()
  end_time = data.time.max()
  graphs = draw_all(data, call)

  pmap = plot_map(data, call)
  title = f'Antenna Analysis for {call}'

  content = template.render(call=call, title=title, antenna=antenna, data=data,
                            start=start_time, end=end_time, map=pmap,
                            graphs=graphs, date=datetime.now(UTC))

  with output.open(mode="w", encoding="utf-8") as fout:
    fout.write(content)
    logging.info('Write file: %s', output)


@log_calls
def gen_filename(call, days, bands):
  call = call.translate(TRANS)
  bands = ''.join(b.rstrip('m') for b in bands)
  bands = bands.replace('0', '')
  created = int(time.time() / 1800)
  combined_string = '|'.join([str(p) for p in (call, days, bands, created)])
  hash_object = hashlib.sha256(combined_string.encode())
  hash_bytes = hash_object.digest()
  short_hash = base64.urlsafe_b64encode(hash_bytes).decode('utf-8')
  short_hash = ''.join(c for c in short_hash if c.isalnum())
  filename = f'{call}-{bands}-{short_hash[:8]}'.upper()
  return f'{filename}.html'


def fetch_and_filter(call, days, bands):
  """Fetch WSPR data and apply band filters.
  Args:
    call: str
    days: int
    bands: list[str] e.g. ['40m', '20m']
  Raises on failure.
  """
  bands = [b.lower() for b in bands]
  _days = f"{days}d"
  df = get_data(call, _days)

  if df.empty:
    raise ValueError(f"No data found for {call}")

  if 'all' not in bands:
    df = df[df['band'].isin(bands)]
    if df.empty:
      raise ValueError(f"No data found for {call} on bands {bands}")

  value_counts = df['band'].value_counts()
  df = df[df['band'].isin(value_counts[value_counts > MIN_ROWS].index)]

  if df.empty:
    raise ValueError(f"No valid data found for {call} band {', '.join(bands)}")
  return df


def run_call(call, days, antenna, bands, outfile):
  try:
    df = fetch_and_filter(call, days, bands)
  except ValueError as err:
    logging.error(err)
    return

  logging.info("%d records", df.shape[0])

  try:
    render_html(df, outfile, call, antenna)
  except Exception as err:  # pylint: disable=broad-exception-caught
    logging.error("Failed to process %s: %s", call, err)


def main():
  parser = argparse.ArgumentParser(description='Antenna Analysis Tool')
  parser.add_argument('-c', '--call', required=True, help='Hamradio callsign')
  parser.add_argument('-b', '--bands', nargs='+', default=['All'], action=BandValidator,
                      help="List of bands to process (use 'all' for all bands)")
  parser.add_argument('-d', '--days', type=int, default=2,
                      help='Number of days (default: %(default)s)')
  parser.add_argument('-f', '--output', type=Path, required=True,
                      help='Output file')
  parser.add_argument('-a', '--antenna', default="Cloud Slayer", help='Antenna name')
  opts = parser.parse_args()

  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

  if opts.output.suffix != '.html':
    logging.error('Invalid file name "%s", the extention must me ".html"', opts.output)
    return os.EX_OSFILE

  run_call(opts.call, opts.days, opts.antenna, opts.bands, opts.output)
  return os.EX_OK


if __name__ == "__main__":
  main()
