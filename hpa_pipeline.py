"""HPA example pipeline: from the four TIFFs of one field of view (FOV) to single-cell crops.

For every FOV in INPUT_FOLDER it runs these steps:
  1  segment nuclei and cells with the HPA Cell Segmentator
  2  leave out cells at the FOV border or too small (optional), cut a 1024x1024 crop per cell
  3  rescale each crop to the KOLF2.1J pixel size, keep the central 512x512 window, convert to 8-bit
  4  quality control of the 512x512 crops (qc.py)

Input, for each FOV: <fov>_blue.tif (nucleus), _green.tif (protein of interest), _red.tif
(microtubules), _yellow.tif (endoplasmic reticulum). 8- or 16-bit, one channel per file. The pixel
size is read from the TIFF resolution tags.

Output in OUTPUT_FOLDER:
  segmentation/<fov>_cellmask.png                    labeled cell mask: pixel value = cell number
  segmentation/<fov>_overview.png                    the segmentation, for viewing (README.md)
  crops/<fov>_cell<N>_{nucleus,protein,microtubules,er,mask}.png   cells that pass QC
  removed/...                                        same files, for cells removed by QC
  qc.csv                                             one row per cropped cell, with QC results

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
from scipy.ndimage import grey_dilation
from skimage.measure import regionprops
from skimage.segmentation import clear_border, find_boundaries

import qc

# ---- settings

INPUT_FOLDER = "examples/hpa/input"
OUTPUT_FOLDER = "output/hpa"                 # compare with examples/hpa/output

SEGMENTATOR_FOLDER = "HPACellSegmentatorPortable"   # checkout with models/ inside, see README.md
DEVICE = "cpu"                                        # or "cuda": faster, masks can differ slightly

TARGET_UM_PER_PX = 0.0941867     # pixel size of the KOLF2.1J crops

FILTER_CELLS = True              # step 2: leave out cells at the FOV border and small cells
BORDER_MARGIN = 4                # px: a nucleus this close to the FOV edge counts as at the border
MIN_CELL_AREA = 2500             # px: smallest cell kept
MIN_NUCLEUS_AREA = 1000          # px: smallest nucleus kept

CHANNELS = {"blue": "nucleus", "green": "protein", "red": "microtubules", "yellow": "er"}


# ---- input

def read_fov(fov):
    """The four channels of a FOV, and its pixel size in um from the TIFF resolution tags."""
    images, resolutions = {}, set()
    for color in CHANNELS:
        with tifffile.TiffFile(f"{INPUT_FOLDER}/{fov}_{color}.tif") as tif:
            page = tif.pages[0]
            images[color] = page.asarray()
            resolutions.add((page.tags["XResolution"].value, page.tags["ResolutionUnit"].value))
    if len(resolutions) != 1:
        raise ValueError(f"{fov}: the four channels have different resolution tags")
    (numerator, denominator), unit = resolutions.pop()
    if unit != 3:                                          # 3 = pixels per centimeter
        raise ValueError(f"{fov}: expected the resolution in pixels per centimeter")
    return images, 1e4 * denominator / numerator


# ---- step 1: segmentation

def load_segmentator():
    sys.path.insert(0, SEGMENTATOR_FOLDER)
    from hpacellseg.cellsegmentator import CellSegmentator
    return CellSegmentator(f"{SEGMENTATOR_FOLDER}/models/dpn_unet_nuclei_v1.pth",
                           f"{SEGMENTATOR_FOLDER}/models/dpn_unet_cell_3ch_v1.pth",
                           scale_factor=0.25, device=DEVICE, padding=True,
                           multi_channel_model=True)


def segment(segmentator, images):
    """Labeled nucleus and cell masks. The protein channel is not used.

    Nuclei are segmented on the nucleus channel, cells on the microtubule, ER and nucleus channels.
    Both run at 1/4 scale. Cell N has nucleus N.
    """
    from hpacellseg.utils import label_cell
    nuclei = segmentator.pred_nuclei([images["blue"]])
    cells = segmentator.pred_cells([[images["red"]], [images["yellow"]], [images["blue"]]])
    nucleus_mask, cell_mask = label_cell(nuclei[0], cells[0])
    return nucleus_mask.astype(np.uint16), cell_mask.astype(np.uint16)


# ---- step 2: which cells, and their 1024x1024 crops

def cells_to_keep(cell_mask, nucleus_mask):
    """Labels of the cells that pass the three filters.

    - the nucleus must not be within BORDER_MARGIN px of the FOV edge. The nucleus mask is made at
      1/4 scale and enlarged 4x, so nucleus edges are 0, 4, 8, ... px from the FOV edge;
    - the cell must have at least MIN_CELL_AREA pixels;
    - the nucleus must have at least MIN_NUCLEUS_AREA pixels.
    A cell that has no nucleus with its label is not removed by the two nucleus rules.
    """
    cells = set(np.unique(cell_mask)) - {0}
    nuclei = set(np.unique(nucleus_mask)) - {0}
    away_from_border = set(np.unique(clear_border(nucleus_mask.astype(np.int32),
                                                  buffer_size=BORDER_MARGIN)))
    cell_area = np.bincount(cell_mask.ravel())
    nucleus_area = np.bincount(nucleus_mask.ravel())

    keep = set()
    for n in cells:
        if n in nuclei and n not in away_from_border:
            continue
        if cell_area[n] < MIN_CELL_AREA:
            continue
        if n in nuclei and nucleus_area[n] < MIN_NUCLEUS_AREA:
            continue
        keep.add(n)
    return keep


def crop_1024(image, cy, cx):
    """The 1024x1024 window of image centered on (cy, cx). Where it runs past the FOV it is 0.

    As in the challenge crops: when the window reaches the bottom or the
    right edge, the last row or column of the FOV is also set to 0.
    """
    h, w = image.shape
    y1, x1, y2, x2 = cy - 512, cx - 512, cy + 512, cx + 512
    pad_top, pad_left = max(0, -y1), max(0, -x1)
    pad_bottom, pad_right = max(0, y2 - (h - 1)), max(0, x2 - (w - 1))
    inside = image[max(0, y1):min(y2, h - 1), max(0, x1):min(x2, w - 1)]
    return np.pad(inside, ((pad_top, pad_bottom), (pad_left, pad_right)))


# ---- step 3: rescale and take the central 512x512 window

def rescale_and_crop_512(crop, source_um_per_px, is_mask=False):
    """Rescale a 1024 crop to TARGET_UM_PER_PX and keep the central 512 window, 8-bit.

    The new size is one of the two whole numbers nearest to 1024 * scale: the one that puts the cell
    center (crop pixel 512) closest to the center of output pixel 256. Area interpolation runs on
    the source values; the 8-bit conversion comes after it. 16-bit values are mapped linearly over
    the full range (v * 255 / 65535); 8-bit values are kept.
    """
    scale = source_um_per_px / TARGET_UM_PER_PX         # 0.850 for the usual HPA pixel size
    center = 512.5                                       # continuous position of crop pixel 512
    best = None
    for size in (int(np.floor(1024 * scale)), int(np.ceil(1024 * scale))):
        start = int(round(center * size / 1024 - 0.5)) - 256
        error = abs(center * size / 1024 - start - 256.5)
        if best is None or error < best[0]:
            best = (error, size, start)
    _, size, start = best

    resized = cv2.resize(crop.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    window = resized[start:start + 512, start:start + 512]

    if is_mask:                                          # binary mask: 0 outside, 255 inside
        return np.where(window > 127, 255, 0).astype(np.uint8)
    if crop.dtype == np.uint16:
        window = window * (255.0 / 65535.0)
    return np.rint(np.clip(window, 0, 255)).astype(np.uint8)


# ---- segmentation overview

def segmentation_overview(nucleus, microtubules, cell_mask, rows):
    """A picture of the segmentation for viewing, the same in kolf_pipeline.py.

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


# ---- one FOV

def process_fov(segmentator, fov):
    """Steps 1-4 for one FOV. Returns one QC row per cropped cell."""
    images, source_um_per_px = read_fov(fov)

    nucleus_mask, cell_mask = segment(segmentator, images)
    cv2.imwrite(f"{OUTPUT_FOLDER}/segmentation/{fov}_cellmask.png", cell_mask)

    keep = cells_to_keep(cell_mask, nucleus_mask) if FILTER_CELLS else None
    rows = []
    for cell in regionprops(cell_mask):
        if keep is not None and cell.label not in keep:
            continue
        # the crop is centered on the center of the cell's bounding box
        y1, x1, y2, x2 = cell.bbox
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2

        # channels: unmasked, source bit depth until the 8-bit conversion in step 3
        crops = {CHANNELS[color]: rescale_and_crop_512(crop_1024(image, cy, cx), source_um_per_px)
                 for color, image in images.items()}
        # mask: this cell only, dilated 7x7 so it reaches 3 px beyond the segmented outline
        mask = grey_dilation((cell_mask == cell.label).astype(np.uint8), size=7) * 255
        crops["mask"] = rescale_and_crop_512(crop_1024(mask, cy, cx), source_um_per_px,
                                             is_mask=True)

        # step 4: quality control
        result = qc.hpa_qc(crops)
        folder = "crops" if result["kept"] else "removed"
        name = f"{fov}_cell{cell.label}"
        for channel, crop in crops.items():
            cv2.imwrite(f"{OUTPUT_FOLDER}/{folder}/{name}_{channel}.png", crop)

        rows.append({"fov": fov, "cell": cell.label, "image_id": name,
                     "source_um_per_px": source_um_per_px, "crop_center_y": cy,
                     "crop_center_x": cx, **result})
    cv2.imwrite(f"{OUTPUT_FOLDER}/segmentation/{fov}_overview.png",
                segmentation_overview(images["blue"], images["red"], cell_mask, rows))
    return rows


if __name__ == "__main__":
    for folder in ("segmentation", "crops", "removed"):
        shutil.rmtree(f"{OUTPUT_FOLDER}/{folder}", ignore_errors=True)
        os.makedirs(f"{OUTPUT_FOLDER}/{folder}")

    segmentator = load_segmentator()
    fovs = sorted(os.path.basename(p)[:-len("_blue.tif")]
                  for p in glob.glob(f"{INPUT_FOLDER}/*_blue.tif"))
    rows = []
    for i, fov in enumerate(fovs, 1):
        fov_rows = process_fov(segmentator, fov)
        rows += fov_rows
        print(f"{i}/{len(fovs)} {fov}: {len(fov_rows)} cells, "
              f"{sum(r['kept'] for r in fov_rows)} pass QC")

    if not rows:
        sys.exit(f"no cells found in {INPUT_FOLDER}")
    with open(f"{OUTPUT_FOLDER}/qc.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
