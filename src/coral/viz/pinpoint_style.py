"""Shared visual language for the Pin Point figures.

One palette and one set of layer functions, so every figure in the chapter reads as the same
drawing. The order below is the drawing order: terrain, then water, then the systems built on
it, then the hazard, then the intervention. A reader who learns the first figure can read all
of them.

Colours are muted so that flood depth and interventions, which carry the result, are the only
saturated things on the page.

    from coral.viz.pinpoint_style import PALETTE, base_map, add_layer, add_scalebar
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

PALETTE = {
    "land":       "#EDE7DD",   # dry ground
    "terrain":    "#8C8478",   # hillshade ink
    "water":      "#9CC3D5",   # open water and channels
    "marsh":      "#A9BE8E",   # NWI wetland
    "building":   "#3D3A36",   # FEMA structures
    "road":       "#C9C2B6",
    "pipe":       "#6E7B8B",   # SAGIS conduits
    "inlet":      "#4A5A6A",   # inlets, manholes
    "outfall":    "#B5651D",   # outfalls, tide gates
    "flood":      "#2C6FA6",   # modelled flood depth (sequential from here)
    "intervention": "#C0392B", # the edit the user draws
    "text":       "#2B2B2B",
    "muted":      "#8A8A8A",
}

# Sequential ramp for depth, light to dark, starting near the water colour.
FLOOD_CMAP = ["#DCEAF3", "#B4D2E5", "#7FB2D3", "#4A8CBF", "#2C6FA6", "#1B4F7E"]


def make_flood_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("pinpoint_flood", FLOOD_CMAP)


def extent_of(h):
    ny, nx, cs = h["nrows"], h["ncols"], h["cellsize"]
    return (h["xllcorner"], h["xllcorner"] + nx * cs,
            h["yllcorner"], h["yllcorner"] + ny * cs)


def hillshade(z, az=315.0, alt=45.0):
    dy, dx = np.gradient(np.nan_to_num(z, nan=float(np.nanmin(z))))
    slope = np.pi / 2 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    a, z0 = np.radians(az), np.radians(alt)
    return (np.sin(z0) * np.sin(slope)
            + np.cos(z0) * np.cos(slope) * np.cos(a - np.pi / 2 - aspect))


def base_map(ax, dem, h, *, sea_level=0.81, hill=True, zorder=0, marsh=None):
    """Terrain and water. Every figure starts here, so the geography is constant.

    `marsh` is an optional boolean grid of tidal wetland. Nearly all of the marsh sits below
    the waterline, so without it the marsh platform is painted as open water and a green
    wetland layer drawn on top just reads as a channel. Passing it splits the two classes:
    open water is below sea level and NOT marsh.
    """
    ext = extent_of(h)
    ax.imshow(np.where(np.isfinite(dem), 1.0, np.nan), extent=ext, origin="upper",
              cmap=_solid(PALETTE["land"]), vmin=0, vmax=1, zorder=zorder)
    if hill:
        ax.imshow(hillshade(dem), extent=ext, origin="upper", cmap="Greys_r",
                  alpha=0.30, zorder=zorder + 0.1)
    below = np.isfinite(dem) & (dem <= sea_level)
    water = below & ~marsh if marsh is not None else below
    ax.imshow(np.where(water, 1.0, np.nan), extent=ext, origin="upper",
              cmap=_solid(PALETTE["water"]), vmin=0, vmax=1, zorder=zorder + 0.2)
    if marsh is not None:
        ax.imshow(np.where(marsh & np.isfinite(dem), 1.0, np.nan), extent=ext,
                  origin="upper", cmap=_solid(PALETTE["marsh"]), vmin=0, vmax=1,
                  zorder=zorder + 0.25)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(PALETTE["muted"]); s.set_linewidth(0.6)
    return ext


def _solid(hexcolor):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("_s", [hexcolor, hexcolor])


def add_vector(ax, path, *, color, lw=0.6, alpha=1.0, ms=2.0, zorder=3, clip=None,
               facecolor=None, label=None):
    """Draw a geojson layer. Points, lines and polygons are handled the same way, so a caller
    does not need to know which a SAGIS layer is. Returns the number of features drawn."""
    try:
        import geopandas as gpd
    except ImportError:
        return 0
    if not Path(path).exists():
        return 0
    g = gpd.read_file(path)
    if g.empty:
        return 0
    g = g.to_crs("EPSG:4326")
    if clip is not None:
        w, e, s, n = clip
        g = g.cx[w:e, s:n]
        if g.empty:
            return 0
    kind = g.geom_type.iloc[0]
    if "Point" in kind:
        ax.scatter(g.geometry.x, g.geometry.y, s=ms, c=color, linewidths=0,
                   alpha=alpha, zorder=zorder, label=label)
    elif "Line" in kind:
        g.plot(ax=ax, color=color, linewidth=lw, alpha=alpha, zorder=zorder, label=label)
    else:
        g.plot(ax=ax, facecolor=facecolor or color, edgecolor="none",
               alpha=alpha, zorder=zorder, label=label)
    return len(g)


def add_scalebar(ax, ext, *, km=1.0, y_frac=0.06, x_frac=0.06, color=None):
    """Scale bar in kilometres. Length is converted through the latitude at the bar."""
    color = color or PALETTE["text"]
    w, e, s, n = ext
    lat = s + (n - s) * y_frac
    deg = km * 1000.0 / (111320.0 * np.cos(np.radians(lat)))
    x0 = w + (e - w) * x_frac
    ax.plot([x0, x0 + deg], [lat, lat], color=color, lw=2.2, solid_capstyle="butt", zorder=20)
    ax.text(x0 + deg / 2, lat + (n - s) * 0.012, f"{km:g} km", ha="center", va="bottom",
            fontsize=7, color=color, zorder=20)


def add_north(ax, ext, *, x_frac=0.94, y_frac=0.90, color=None):
    color = color or PALETTE["text"]
    w, e, s, n = ext
    x, y = w + (e - w) * x_frac, s + (n - s) * y_frac
    ax.annotate("N", xy=(x, y), xytext=(x, y - (n - s) * 0.045), ha="center",
                fontsize=8, color=color, zorder=20,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0))


def add_callout(ax, xy, text, *, dxy=(0.05, 0.05), ext=None, fontsize=7.5, color=None):
    """Leader line and label. Offsets are fractions of the axis span, so a callout keeps its
    look when the panel is resized."""
    color = color or PALETTE["text"]
    if ext is not None:
        w, e, s, n = ext
        tx, ty = xy[0] + (e - w) * dxy[0], xy[1] + (n - s) * dxy[1]
    else:
        tx, ty = xy[0] + dxy[0], xy[1] + dxy[1]
    ax.annotate(text, xy=xy, xytext=(tx, ty), fontsize=fontsize, color=color, zorder=21,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.7, shrinkA=0, shrinkB=2))


def panel_title(ax, letter, title, subtitle=None, *, y=1.03):
    """Panel label: a letter, a short title, one line of explanation under it.

    The subtitle sits below the title, not above, so a long one grows away from the panel
    instead of into the figure title.
    """
    ax.set_title("", loc="left")
    ax.text(0.0, y + 0.045, letter, transform=ax.transAxes, fontsize=14, fontweight="bold",
            color=PALETTE["text"], va="bottom")
    ax.text(0.075, y + 0.045, title, transform=ax.transAxes, fontsize=11,
            color=PALETTE["text"], va="bottom")
    if subtitle:
        ax.text(0.0, y, subtitle, transform=ax.transAxes, fontsize=7.8,
                color=PALETTE["muted"], va="bottom")
