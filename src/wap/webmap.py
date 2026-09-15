#! /usr/bin/env python
# vim:fenc=utf-8
#
# Copyright © 2026 fred <github-fred@hidzz.com>
#
# Distributed under terms of the BSD 3-Clause license.

import re
from dataclasses import dataclass

import folium
from folium.plugins import Geocoder, HeatMap, MarkerCluster


@dataclass(slots=True)
class Spot:
  # pylint: disable=too-many-instance-attributes
  call: str
  band: str
  lat: float
  lon: float
  distance: int
  count: int
  snr_min: int
  snr_mean: int
  snr_max: int

  def __str__(self):
    template = f"""<b>{self.call}</b>
      Band: <b>{self.band}</b>
      Distance: <b>{self.distance}km</b>
      Number of spots: <b>{self.count}</b>
      SNR min: <b>{self.snr_min}dB</b>
      SNR average: <b>{self.snr_mean}dB</b>
      SNR max: <b>{self.snr_max}dB</b>
      """
    return format_tooltip(template)


def format_tooltip(msg):
  def _nl(match):
    if match.group().startswith('\n'):
      return '<br>'
    return '&nbsp;'

  return re.sub(r' +|\n +', _nl, msg)


def display_markers(wmap, spots):
  # color = '#e377c2' color = '#7f7f7f' color = '#bcbd22' color = '#17becf'
  marker_cluster = MarkerCluster().add_to(wmap)
  for _, row in spots.iterrows():
    spot = Spot(*row.values)
    match row['band']:
      case '160m':
        color = 'darkpurple'
      case '80m':
        color = 'red'
      case '40m':
        color = 'blue'
      case '20m':
        color = 'green'
      case '15m':
        color = 'darkblue'
      case '10m':
        color = 'pink'
      case _:
        color = 'beige'

    icon = 'tower-broadcast'
    try:
      folium.Marker(
        location=[spot.lat, spot.lon],
        icon=folium.Icon(color=color, prefix='fa', icon=icon),
        popup=str(spot)
      ).add_to(marker_cluster)
    except ValueError:
      print('ValueError', spot.lat, spot.lon, spot.call)


def plot_map(data, call):
  pwr = data['power'].iloc[0].mean()
  grid = data['tx_loc'].iloc[0]
  latlon = data[['tx_lat', 'tx_lon']].iloc[0].values
  spots = data[['rx_sign', 'rx_lat', 'rx_lon', 'distance', 'band', 'snr']].copy()
  spots = (
    spots
    .groupby(['rx_sign', 'band'])
    .agg(
      rx_lat=('rx_lat', 'first'),
      rx_lon=('rx_lon', 'first'),
      distance=('distance', 'first'),
      count=('snr', 'size'),
      snr_min=('snr', 'min'),
      snr_mean=('snr', 'mean'),
      snr_max=('snr', 'max'),
    )
    .reset_index()
  )
  spots['distance'] = spots['distance'].round().astype(int)
  spots['snr_mean'] = spots['snr_mean'].round().astype(int)

  attr = ('&copy; <a href="https://www.OpenStreetMap.org/copyright">OpenStreetMap</a> '
          'contributors Fred <a href="https://qrz.com/db/W6BSD">W6BSD</a>')

  wmap = folium.Map(location=latlon, zoom_start=3, tiles=None)

  folium.TileLayer(
    tiles='https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    name='OpenStreet Map',
    attr=attr
  ).add_to(wmap)

  Geocoder(add_marker=False, zoom=8, provider="photon").add_to(wmap)
  folium.Marker(
    latlon,
    icon=folium.Icon(prefix='fa', icon='walkie-talkie', color='red'),
    popup=format_tooltip(
      f'<b>{call}</b> Transmitter\n'
      f'Grid: <b>{grid}</b>\n'
      f'Power: <b>{pwr:.0f}dBm</b>'),
  ).add_to(wmap)

  for band in sorted(spots.band.unique()):
    layer = folium.FeatureGroup(name=band)
    selection = spots[spots.band == band]
    HeatMap(data=selection[['rx_lat', 'rx_lon', 'count']], radius=17).add_to(layer)
    display_markers(layer, selection)
    layer.add_to(wmap)

  folium.LayerControl().add_to(wmap)
  return wmap._repr_html_()  # pylint: disable=protected-access
