"""KOLF2.1J example pipeline: from the four TIFFs of one z-plane to single-cell crops.

The KOLF2.1J images are confocal z-stacks. For each stack, the sharpest plane was picked (see
README.md); the example inputs are those picked planes. For every image in INPUT_FOLDER this script
runs these steps:
  1  subtract the background of each channel (rolling ball)
  2  segment nuclei and cells with the HPA Cell Segmentator, on the raw plane with a per-image gain
  3  per cell: a 512x512 crop window centered on the cell's bounding box
  4  scale each channel to 8-bit with the upper limits in UPPER_LIMITS (per well, see README.md)
  5  quality control (qc.py), then write the crops

Input, for each image: <image>_blue.tif (nucleus), _green.tif (protein of interest), _red.tif
(microtubules), _yellow.tif (endoplasmic reticulum), 16-bit, and its rows in UPPER_LIMITS.

Output in OUTPUT_FOLDER:
  segmentation/<image>_cellmask.png                    labeled cell mask: pixel value = cell number
  segmentation/<image>_overview.png                    the segmentation, for viewing (README.md)
  crops/<image>_cell<N>_{nucleus,protein,microtubules,er,mask}.png   cells that pass QC
  removed/...                                          same files, for cells removed by QC
  qc.csv                                               one row per cropped cell, with QC results

Needs numpy, scipy, scikit-image, opencv-python, tifffile, torch, and the HPA Cell Segmentator
(see README.md).
"""

import csv
import glob
import os
import shutil
import sys

import cv2
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import grey_dilation
from skimage.restoration import ellipsoid_kernel, rolling_ball
from skimage.segmentation import find_boundaries
from skimage.transform import resize

import qc

# ---- settings

INPUT_FOLDER = "examples/kolf/input"
UPPER_LIMITS = "examples/kolf/input/upper_limits.csv"
OUTPUT_FOLDER = "output/kolf"            # compare with examples/kolf/output

SEGMENTATOR_FOLDER = "HPACellSegmentatorPortable"   # checkout with models/ inside, see README.md
DEVICE = "cpu"                                        # or "cuda": faster, masks can differ slightly

RADIUS = 500                     # rolling-ball radius in px: larger than a cell
DOWNSCALE = 8                    # the background is estimated on a 1/8 copy of the image
CROP = 512

# step 2: gain for the segmentation input only. The raw KOLF2.1J planes are much dimmer than HPA
# images; each channel is scaled so that its 99th percentile lands on the median 99th percentile of
# the HPA crops (8-bit values 167, 164 and 154, times 256)
SEGMENTATION_P99 = {"blue": 167 * 256, "red": 164 * 256, "yellow": 154 * 256}

CHANNELS = {"blue": "nucleus", "green": "protein", "red": "microtubules", "yellow": "er"}


# ---- input

def read_image(name):
    """The four raw 16-bit channels of one image, keyed by color."""
    return {color: tifffile.imread(f"{INPUT_FOLDER}/{name}_{color}.tif") for color in CHANNELS}


def read_upper_limits(path):
    """{image: {"upper": {channel: value}, "upper_qc": {channel: value}}}."""
    limits = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            entry = limits.setdefault(row["image"], {"upper": {}, "upper_qc": {}})
            entry["upper"][row["channel"]] = float(row["upper"])
            entry["upper_qc"][row["channel"]] = float(row["upper_qc"])
    return limits


# ---- step 1: background subtraction

def subtract_background(image):
    """Rolling-ball background, estimated on a downscaled copy and resized back, subtracted.

    Keep RADIUS and DOWNSCALE fixed across a dataset: a coarser DOWNSCALE estimates a slightly
    higher background.
    """
    small = resize(image, (image.shape[0] // DOWNSCALE, image.shape[1] // DOWNSCALE),
                   order=1, preserve_range=True, anti_aliasing=True)
    spatial = max(1, int(round(RADIUS / DOWNSCALE)))
    kernel = ellipsoid_kernel((2 * spatial + 1, 2 * spatial + 1), 2 * RADIUS)
    background = resize(rolling_ball(small, kernel=kernel), image.shape,
                        order=1, preserve_range=True)
    subtracted = image.astype(np.float32) - background
    np.clip(subtracted, 0, 65535, out=subtracted)
    return subtracted.astype(np.uint16)


# ---- step 2: segmentation

def segmentation_input(image, color):
    """Linear gain so the image's 99th percentile lands on SEGMENTATION_P99. Segmentation only."""
    gain = SEGMENTATION_P99[color] / max(float(np.percentile(image, 99)), 1.0)
    return np.clip(np.rint(image.astype(np.float64) * gain), 0, 65535).astype(np.uint16)


def load_segmentator():
    sys.path.insert(0, SEGMENTATOR_FOLDER)
    from hpacellseg.cellsegmentator import CellSegmentator
    return CellSegmentator(f"{SEGMENTATOR_FOLDER}/models/dpn_unet_nuclei_v1.pth",
                           f"{SEGMENTATOR_FOLDER}/models/dpn_unet_cell_3ch_v1.pth",
                           scale_factor=0.25, device=DEVICE, padding=True,
                           multi_channel_model=True)


def segment(segmentator, raw):
    """Labeled nucleus and cell masks from the raw channels. The protein channel is not used.

    Nuclei are segmented on the nucleus channel, cells on the microtubule, ER and nucleus channels,
    each after segmentation_input(). Both run at 1/4 scale. Cell N has nucleus N.

    The HPA Cell Segmentator is used because the HPA pipeline needs it anyway. It can merge touching
    cells. Newer models, such as Cellpose-SAM and Cellpose-DINO, may segment your images better,
    alone or combined (https://github.com/MouseLand/cellpose; see README.md).
    """
    from hpacellseg.utils import label_cell
    red, yellow, blue = (segmentation_input(raw[c], c) for c in ("red", "yellow", "blue"))
    nuclei = segmentator.pred_nuclei([blue])
    cells = segmentator.pred_cells([[red], [yellow], [blue]])
    nucleus_mask, cell_mask = label_cell(nuclei[0], cells[0])
    return nucleus_mask.astype(np.int32), cell_mask.astype(np.int32)


# ---- step 3: crop windows

def plan_crops(cell_mask):
    """One row per cell whose 512x512 window, centered on its bounding box, fits in the image.

    Cells whose window would run past the edge of the image are not cropped.
    """
    h, w = cell_mask.shape
    rows = []
    for label, box in enumerate(ndi.find_objects(cell_mask), start=1):
        if box is None:
            continue
        (y0, y1), (x0, x1) = (box[0].start, box[0].stop), (box[1].start, box[1].stop)
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2      # bounding-box center, as in hpa_pipeline.py
        if CROP // 2 <= cy <= h - CROP // 2 and CROP // 2 <= cx <= w - CROP // 2:
            rows.append({"cell": label, "crop_center_y": cy, "crop_center_x": cx})
    return rows


def crop_mask(cell_mask, label, window):
    """The crop's mask, as in hpa_pipeline.py: the cell's label, dilated with a 7x7 gray dilation so
    it reaches 3 px beyond the segmented outline. 255 inside, 0 outside."""
    return (grey_dilation((cell_mask == label).astype(np.uint8), size=7) * 255)[window]


# ---- step 4: 8-bit

def to_8bit(image, upper):
    """Linear map: 0 stays 0, `upper` becomes 255, brighter values are clipped."""
    scaled = image.astype(np.float32) * (255.0 / max(upper, 1))
    np.clip(scaled, 0, 255, out=scaled)
    return np.rint(scaled).astype(np.uint8)


# ---- segmentation overview

def segmentation_overview(nucleus, microtubules, cell_mask, rows):
    """A picture of the segmentation for viewing, the same in hpa_pipeline.py.

    Half resolution. Nucleus in blue and microtubules in red, as in the crops (KOLF2.1J: after
    background subtraction), each scaled to its 99.9th percentile. Each cell is outlined: green if
    it was cropped and passed QC, magenta if QC removed it, gray if it was not cropped. Cropped
    cells carry their number at the crop center.
    """
    def scaled(image):                                   # half resolution keeps the file small
        half = cv2.resize(image.astype(np.float32), (image.shape[1] // 2, image.shape[0] // 2),
                          interpolation=cv2.INTER_AREA)
        top = max(float(np.percentile(half, 99.9)), 1.0)
        return np.clip(half * (255.0 / top), 0, 255).astype(np.uint8)

    h, w = (n // 2 * 2 for n in cell_mask.shape)          # even size, so all half-size images match
    labels = cell_mask[:h:2, :w:2]
    picture = np.dstack([scaled(nucleus), np.zeros(labels.shape, np.uint8), scaled(microtubules)])
    colors = np.full((int(cell_mask.max()) + 1, 3), 128, np.uint8)        # BGR, gray
    for row in rows:
        colors[row["cell"]] = (0, 255, 0) if row["kept"] else (255, 0, 255)
    outline = grey_dilation(np.where(find_boundaries(labels, mode="inner"), labels, 0), size=2)
    picture[outline > 0] = colors[outline[outline > 0]]
    for row in rows:
        position = (row["crop_center_x"] // 2 - 12, row["crop_center_y"] // 2 + 8)
        for color, thickness in (((0, 0, 0), 5), ((255, 255, 255), 2)):     # black edge, white text
            cv2.putText(picture, str(row["cell"]), position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color,
                        thickness)
    return picture


# ---- one image

def process_image(segmentator, name, limits):
    """Steps 1-5 for one image. Returns one QC row per cropped cell."""
    raw = read_image(name)

    # step 1: background subtraction
    background_subtracted = {CHANNELS[c]: subtract_background(raw[c]) for c in CHANNELS}

    # step 2: segmentation, on the raw plane
    nucleus_mask, cell_mask = segment(segmentator, raw)
    cv2.imwrite(f"{OUTPUT_FOLDER}/segmentation/{name}_cellmask.png", cell_mask.astype(np.uint16))

    # steps 3-4: crop windows and 8-bit images
    upper, upper_qc = limits[name]["upper"], limits[name]["upper_qc"]
    qc_images = {ch: to_8bit(im, upper_qc[ch]) for ch, im in background_subtracted.items()}
    images = {ch: to_8bit(im, upper[ch]) for ch, im in background_subtracted.items()}

    rows = plan_crops(cell_mask)
    windows = {r["cell"]: (slice(r["crop_center_y"] - CROP // 2, r["crop_center_y"] + CROP // 2),
                           slice(r["crop_center_x"] - CROP // 2, r["crop_center_x"] + CROP // 2))
               for r in rows}
    masks = {label: crop_mask(cell_mask, label, window) for label, window in windows.items()}

    # step 5: quality control
    rows = qc.kolf_qc(rows, cell_mask, nucleus_mask, qc_images, masks)
    for row in rows:
        label = row["cell"]
        image_id = f"{name}_cell{label}"
        crops = {ch: im[windows[label]] for ch, im in images.items()}
        crops["mask"] = masks[label]
        folder = "crops" if row["kept"] else "removed"
        for channel, crop in crops.items():
            cv2.imwrite(f"{OUTPUT_FOLDER}/{folder}/{image_id}_{channel}.png", crop)
        row.update({"image": name, "image_id": image_id})
    cv2.imwrite(f"{OUTPUT_FOLDER}/segmentation/{name}_overview.png",
                segmentation_overview(background_subtracted["nucleus"],
                                      background_subtracted["microtubules"], cell_mask, rows))
    return [{"image": r.pop("image"), "image_id": r.pop("image_id"), **r} for r in rows]


if __name__ == "__main__":
    for folder in ("segmentation", "crops", "removed"):
        shutil.rmtree(f"{OUTPUT_FOLDER}/{folder}", ignore_errors=True)
        os.makedirs(f"{OUTPUT_FOLDER}/{folder}")

    limits = read_upper_limits(UPPER_LIMITS)
    names = sorted(os.path.basename(p)[:-len("_blue.tif")]
                   for p in glob.glob(f"{INPUT_FOLDER}/*_blue.tif"))
    missing = [n for n in names
               if n not in limits or set(limits[n]["upper"]) != set(CHANNELS.values())]
    if missing:
        sys.exit(f"upper limits missing in {UPPER_LIMITS} for: {', '.join(missing)}")

    segmentator = load_segmentator()
    rows = []
    for i, name in enumerate(names, 1):
        image_rows = process_image(segmentator, name, limits)
        rows += image_rows
        print(f"{i}/{len(names)} {name}: {len(image_rows)} cells, "
              f"{sum(r['kept'] for r in image_rows)} pass QC")

    if not rows:
        sys.exit(f"no cells found in {INPUT_FOLDER}")
    with open(f"{OUTPUT_FOLDER}/qc.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
