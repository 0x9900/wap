# WSPR Antenna Performance (WAP)

WAP is a ham radio antenna analysis tool that uses real **WSPR reception data**. Instead of relying only on models or simulations, it looks at how your antenna actually works on the air.

The project aims to help radio amateurs **evaluate, compare, and improve** their antenna setups using clear, data-driven methods.

---

## What WAP Does

WAP takes WSPR spot data and pulls out performance details for your transmitting station. Specifically, it:

* Collects and filters WSPR reception reports
* Groups spots by band, distance, and azimuth
* Removes extreme statistical outliers to reduce propagation noise
* Produces polar plots and other visualizations that reveal antenna performance and radiation characteristics

With this, you can see how your antenna works in different directions and at various distances, using thousands of real reception reports.

---

## Why WSPR?

WSPR is especially useful for antenna analysis because:

* It provides large volumes of standardized reception reports[^1]
* SNR measurements are consistent and comparable
* Reports include precise distance and azimuth information
* Long-term data smooths out short-term propagation effects

By gathering data over time and distance, WAP shows overall antenna behavior instead of just short-term changes in propagation.

---

## Typical Workflow

1. Hams transmit WSPR signals
2. Visit https://wspr.bsdworld.org/ to create plots for the bands and distance ranges you choose
3. WAP gathers WSPR spot data for your call-sign
4. Compare directional patterns between different antennas, setups, or time periods

This process makes WAP helpful for:

* Antenna A/B comparisons
* Evaluating antenna height or orientation changes
* Assessing feedline, balun, or matching changes
* Long-term station performance monitoring

---

## Status

The project is still under active development. Data formats and visualizations might change as the analysis methods improve.

Feedback from experienced radio amateurs is always welcome.

---

## License

BSD 3-Clause License. See the LICENSE file for more details.

---

## Author

WAP is developed by Fred W6BSD, a radio amateur who is passionate about **antenna efficiency, measurement, and real-world performance analysis**.

---

[^1] Unfortunately, it is not geographically evenly distributed.
