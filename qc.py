"""Example quality control of the single-cell crops, for both data sets.

The rules are based on the QC of the challenge data, and their thresholds were set on these two
image sets; check them before using them on other data.

    hpa_qc(crops)                     HPA. Looks only at the five 512x512 PNGs of one cell.
    kolf_qc(cells, ..., masks)        KOLF2.1J. Also uses the full-size masks of the image, and
                                      compares each cell with the other cells of its image.

A cell is removed if any rule applies:

    rule                 HPA  KOLF2.1J
    nucleus_free_mask     x      x     mean nucleus inside the cell mask < 0.2 x outside it
    no_nucleus_in_crop    x      x     nucleus channel of the crop all 0 or 99th percentile <= 2
    faint_reference       x      x     microtubule or ER channel of the crop all 0 or 99th
                                       percentile <= 2
    padded                x            more than 50% of the crop is zero padding
    broken_mask           x            empty mask, or a whole-crop mask with two or more large
                                       nuclei
    small                        x     cell < 2,500 px or nucleus < 1,000 px
    fragmented                   x     the cell's largest connected part holds < 95% of it
    no_nucleus                   x     no nucleus, or < 90% of the nucleus lies inside the cell
    nucleus_dim                  x     own-nucleus mean < 0.2 x the median of the image's cells
    no_marker                    x     cytoplasm microtubule + ER mean < 0.1 x the median of the
                                       image's cells
    nucleus_faint                x     own-nucleus mean < 0.3 x the median of the image's cropped
                                       cells that pass small, fragmented and no_nucleus
    nucleus_small                x     the nucleus covers < 2,000 px of the crop window

HPA cells at the FOV border or too small are left out before cropping (FILTER_CELLS in
hpa_pipeline.py). KOLF2.1J crops never run past the image, so they need no `padded` rule. The HPA
and KOLF2.1J challenge crops computed the 99th percentile in slightly different ways; both are kept.

Needs numpy and scipy.
"""

import numpy as np
from scipy import ndimage as ndi

CROP_SIZE = 512

# both data sets
NUCLEUS_FREE_RATIO = 0.2         # nucleus_free_mask: inside mean < this x outside mean
BLANK_P99 = 2                    # a crop channel with 99th percentile <= this counts as blank

# HPA only
MAX_PAD_FRACTION = 0.5           # padded: more than this fraction of the crop is padding
MIN_NUCLEUS_PX = 3000            # smallest object counted as a nucleus (whole-crop masks only)
LARGE_NUCLEUS_PX = 8000          # smallest large nucleus (whole-crop masks only)

# KOLF2.1J only
MIN_CELL_AREA = 2500             # px: cell smaller -> small
MIN_NUCLEUS_AREA = 1000          # px: nucleus smaller -> small
MIN_LARGEST_COMPONENT = 0.95     # the cell's largest connected part holds less -> fragmented
MIN_NUCLEUS_INSIDE = 0.9         # less of the nucleus lies inside its cell -> no_nucleus
NUCLEUS_DIM_RATIO = 0.2          # own-nucleus mean < this x the image median -> nucleus_dim
MARKER_RATIO = 0.1               # cytoplasm MT + ER mean < this x the image median -> no_marker
NUCLEUS_FAINT_RATIO = 0.3        # own-nucleus mean < this x the image median -> nucleus_faint
MIN_NUCLEUS_MASK_PX = 2000       # nucleus inside the crop window smaller -> nucleus_small

HPA_RULES = ("nucleus_free_mask", "padded", "no_nucleus_in_crop", "faint_reference", "broken_mask")
KOLF_GEOMETRY_RULES = ("small", "fragmented", "no_nucleus")
KOLF_IMAGE_RULES = ("nucleus_free_mask", "nucleus_dim", "no_marker", "no_nucleus_in_crop",
                    "faint_reference", "nucleus_faint", "nucleus_small")


# ---- HPA

def percentile_99(image):
    """99th percentile of an 8-bit image: the lowest value v with at least 99% of pixels <= v."""
    cdf = np.cumsum(np.bincount(image.ravel(), minlength=256))
    return int(np.searchsorted(cdf, cdf[-1] * 0.99))


def padding_fraction(zero):
    """Fraction of the crop that is edge padding.

    `zero` marks pixels that are 0 in all four image channels. Padding is the run of all-zero rows
    and columns counted inwards from each edge of the crop.
    """

    def leading_run(line):
        return len(line) if line.all() else int(np.argmin(line))

    rows, cols = zero.all(axis=1), zero.all(axis=0)
    height = max(0, CROP_SIZE - leading_run(rows) - leading_run(rows[::-1]))
    width = max(0, CROP_SIZE - leading_run(cols) - leading_run(cols[::-1]))
    return 1 - height * width / CROP_SIZE**2


def otsu_threshold(image):
    """Otsu threshold of an image with values in 0..255."""
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    p = hist / hist.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    between = (mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    return np.nanargmax(between)


def count_nuclei(nucleus):
    """Number of nuclei (>= MIN_NUCLEUS_PX) and of large nuclei (>= LARGE_NUCLEUS_PX) in a crop.

    Used only for masks that cover the whole crop, where the mask itself says nothing about how
    many cells it contains. Gaussian smoothing, Otsu threshold (at least 10), opening, hole filling.
    """
    smooth = ndi.gaussian_filter(nucleus.astype(float), 2)
    if smooth.max() < 5:
        return 0, 0
    binary = smooth > max(otsu_threshold(smooth), 10)
    binary = ndi.binary_fill_holes(ndi.binary_opening(binary, iterations=2))
    labels, _ = ndi.label(binary)
    areas = np.bincount(labels.ravel())[1:]
    return int((areas >= MIN_NUCLEUS_PX).sum()), int((areas >= LARGE_NUCLEUS_PX).sum())


def hpa_qc(crops):
    """HPA rules for one cell. `crops`: its five 512x512 uint8 PNGs, keyed nucleus, protein,
    microtubules, er, mask. Returns the measurements, one True/False per rule, and `kept`."""
    nucleus, microtubule = crops["nucleus"], crops["microtubules"]
    er, mask = crops["er"], crops["mask"]
    inside = mask > 0
    zero = (nucleus == 0) & (crops["protein"] == 0) & (microtubule == 0) & (er == 0)
    # pixels that are 0 in all four channels (padding, empty background) do not count as outside
    outside = ~inside & ~zero
    row = {
        "pad_fraction": padding_fraction(zero),
        "mask_area": int(inside.sum()),
        "whole_crop_mask": bool((mask == 255).all()),
        "nucleus_mean_in_mask": float(nucleus[inside].mean()) if inside.any() else np.nan,
        "nucleus_mean_outside_mask": float(nucleus[outside].mean()) if outside.any() else np.nan,
    }
    for name, image in (("nucleus", nucleus), ("microtubules", microtubule), ("er", er)):
        row[f"{name}_p99"] = percentile_99(image)
        row[f"{name}_max"] = int(image.max())
    if row["whole_crop_mask"]:
        row["n_nuclei"], row["n_large_nuclei"] = count_nuclei(nucleus)
    else:
        row["n_nuclei"] = row["n_large_nuclei"] = np.nan

    def blank(name):
        return row[f"{name}_max"] == 0 or row[f"{name}_p99"] <= BLANK_P99

    whole = row["whole_crop_mask"]
    row.update({
        "nucleus_free_mask": bool(
            row["nucleus_mean_in_mask"] < NUCLEUS_FREE_RATIO * row["nucleus_mean_outside_mask"]
        ),
        "padded": row["pad_fraction"] > MAX_PAD_FRACTION,
        "no_nucleus_in_crop": blank("nucleus") or (whole and row["n_nuclei"] == 0),
        "faint_reference": blank("microtubules") or blank("er"),
        "broken_mask": row["mask_area"] == 0 or (whole and row["n_large_nuclei"] >= 2),
    })
    row["kept"] = not any(row[rule] for rule in HPA_RULES)
    return row


# ---- KOLF2.1J

def kolf_geometry(cells, cell_mask, nucleus_mask):
    """Geometry rules on the full-size masks: small, fragmented, no_nucleus.

    touches_border (the cell touches the image edge) is recorded but is not a rule.
    """
    n = int(cell_mask.max())
    area = np.bincount(cell_mask.ravel(), minlength=n + 1)
    nucleus_area = np.bincount(nucleus_mask.ravel(), minlength=n + 1)[:n + 1]
    own = (nucleus_mask > 0) & (nucleus_mask == cell_mask)
    nucleus_inside = np.bincount(nucleus_mask[own], minlength=n + 1)[:n + 1]
    edge = set(np.unique(np.concatenate([cell_mask[0], cell_mask[-1], cell_mask[:, 0],
                                         cell_mask[:, -1]]))) - {0}
    boxes = ndi.find_objects(cell_mask)
    for row in cells:
        label = row["cell"]
        parts, n_parts = ndi.label(cell_mask[boxes[label - 1]] == label)
        largest = np.bincount(parts.ravel())[1:].max() / area[label] if n_parts else 0.0
        inside = nucleus_inside[label] / nucleus_area[label] if nucleus_area[label] else 0.0
        row.update({
            "cell_area": int(area[label]), "nucleus_area": int(nucleus_area[label]),
            "largest_component_frac": float(largest), "nucleus_inside_frac": float(inside),
            "touches_border": label in edge,
            "small": bool(area[label] < MIN_CELL_AREA or nucleus_area[label] < MIN_NUCLEUS_AREA),
            "fragmented": bool(largest < MIN_LARGEST_COMPONENT),
            "no_nucleus": bool(nucleus_area[label] == 0 or inside < MIN_NUCLEUS_INSIDE),
        })


def kolf_image_rules(cells, cell_mask, nucleus_mask, qc_images, masks):
    """Image measurements and rules (all but nucleus_faint), on the 8-bit images at upper_qc."""
    n = int(cell_mask.max())
    labels = np.arange(1, n + 1)
    cytoplasm = np.where(nucleus_mask == cell_mask, 0, cell_mask)
    nucleus_image = qc_images["nucleus"].astype(np.float32)
    nucleus_mean = np.asarray(ndi.mean(nucleus_image, nucleus_mask, labels))
    markers = qc_images["microtubules"].astype(np.float32) + qc_images["er"]
    marker_mean = np.asarray(ndi.mean(markers, cytoplasm, labels))
    image_nucleus, image_marker = np.nanmedian(nucleus_mean), np.nanmedian(marker_mean)
    for row in cells:
        label, cy, cx = row["cell"], row["crop_center_y"], row["crop_center_x"]
        window = (slice(cy - CROP_SIZE // 2, cy + CROP_SIZE // 2),
                  slice(cx - CROP_SIZE // 2, cx + CROP_SIZE // 2))
        inside = masks[label] > 0
        nucleus = qc_images["nucleus"][window]
        mean_in, mean_out = nucleus[inside].mean(), nucleus[~inside].mean()
        row.update({
            "mask_px": int(inside.sum()),
            "nucleus_mask_px": int((nucleus_mask[window] == label).sum()),
            "nucleus_in_out": float(mean_in / mean_out) if mean_out > 0 else np.inf,
            "nucleus_mean": float(nucleus_mean[label - 1]),
            "nucleus_rel": float(nucleus_mean[label - 1] / image_nucleus),
            "marker_rel": float(marker_mean[label - 1] / image_marker),
        })
        for channel, image in qc_images.items():
            crop = image[window]
            row[f"{channel}_p99"] = float(np.percentile(crop, 99))
            row[f"{channel}_max"] = int(crop.max())

        def blank(channel):
            return row[f"{channel}_max"] == 0 or row[f"{channel}_p99"] <= BLANK_P99

        geometry_pass = not any(row[rule] for rule in KOLF_GEOMETRY_RULES)
        row.update({
            "nucleus_free_mask": row["nucleus_in_out"] < NUCLEUS_FREE_RATIO,
            "nucleus_dim": not row["nucleus_rel"] >= NUCLEUS_DIM_RATIO,   # NaN (no nucleus) fails
            "no_marker": not row["marker_rel"] >= MARKER_RATIO,
            "no_nucleus_in_crop": blank("nucleus"),
            "faint_reference": blank("microtubules") or blank("er"),
            "nucleus_small": geometry_pass and row["nucleus_mask_px"] < MIN_NUCLEUS_MASK_PX,
        })


def kolf_qc(cells, cell_mask, nucleus_mask, qc_images, masks):
    """KOLF2.1J rules for the cropped cells of one image. Adds measurements, rules and `kept`.

    cells:      one row per cropped cell, with cell, crop_center_y, crop_center_x
    qc_images:  the four channels of the image in 8-bit at the QC limits (upper_qc); the image
                rules were set at that brightness, so they are not evaluated on the crops themselves
    masks:      {cell: its 512x512 crop mask}

    nucleus_faint compares with the median over the image's cropped cells that pass the geometry
    rules (the challenge data used all images of a well). nucleus_faint and nucleus_small are only
    evaluated for cells that pass the geometry rules; for the other cells they are False.
    """
    kolf_geometry(cells, cell_mask, nucleus_mask)
    kolf_image_rules(cells, cell_mask, nucleus_mask, qc_images, masks)
    passing = [r for r in cells if not any(r[rule] for rule in KOLF_GEOMETRY_RULES)]
    median = np.median([r["nucleus_mean"] for r in passing]) if passing else np.nan
    for r in cells:
        geometry_pass = not any(r[rule] for rule in KOLF_GEOMETRY_RULES)
        r["nucleus_faint"] = geometry_pass and r["nucleus_mean"] < NUCLEUS_FAINT_RATIO * median
        r["kept"] = not any(r[rule] for rule in KOLF_GEOMETRY_RULES + KOLF_IMAGE_RULES)
    return cells
