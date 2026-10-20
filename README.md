# CM4AI/AnVIL Cell Map Prediction DREAM Challenge: example preprocessing pipelines

This repository holds two example pipelines that turn microscope images into single-cell crops like
those of the CM4AI/AnVIL Cell Map Prediction DREAM Challenge. They run on a few example images and
show the main steps: segmentation, cropping, intensity scaling and quality control (QC). They follow
the processing of the challenge data closely, but not in every detail; the main differences are
noted below. Their settings are chosen for these two image sets, so other data need adapting (see
[Adapting to other data](#adapting-to-other-data)).

- **HPA**: confocal immunofluorescence images of the Human Protein Atlas (HPA), subcellular
  section, release v25. One field of view (FOV) is one plane, as four single-channel images.
- **KOLF2.1J**: confocal immunofluorescence z-stacks of the KOLF2.1J induced pluripotent stem
  cell (iPSC) line, from the Cell Maps for Artificial Intelligence (CM4AI) project. The pipeline
  starts from one plane per stack.

Both pipelines produce the same output format: five 512×512 8-bit PNGs per cell, at 0.0941867 µm
per pixel, followed by quality control (QC) of the crops.

| file | contents |
|---|---|
| `<image_id>_nucleus.png` | nucleus channel |
| `<image_id>_protein.png` | protein of interest (antibody staining) |
| `<image_id>_microtubules.png` | microtubules |
| `<image_id>_er.png` | endoplasmic reticulum |
| `<image_id>_mask.png` | mask of the center cell, 255 inside and 0 outside |

`<image_id>` is `<image>_cell<N>`, as in the `image_id` column of `qc.csv`.

## HPA and KOLF2.1J crops at a glance

| | HPA | KOLF2.1J |
|---|---|---|
| source image | one confocal plane, mostly 2048×2048 at 0.0800885 µm per pixel | sharpest plane of a confocal z-stack, 2160×2160 at 0.0941867 µm per pixel |
| resampling | rescaled to 0.0941867 µm per pixel (×0.85) | none |
| background | not subtracted | rolling ball, subtracted |
| 16 → 8 bit | fixed: `v × 255 / 65535` | per well: 0 → 0, the well's upper limit → 255, brighter values clipped |
| segmentation | HPA Cell Segmentator | HPA Cell Segmentator (challenge data: consensus of HPA Cell Segmentator, Cellpose-SAM and Cellpose-DINO) |
| crop | centered on the cell's bounding box; zero-padded where it runs past the FOV | centered on the cell's bounding box; cells whose crop would run past the image are left out |
| mask | the center cell, dilated 3 px (before rescaling) | the center cell, dilated 3 px |
| QC (`qc.py`) | 5 rules, on the crop PNGs only | 10 rules, on the crops, the full-size masks and the other cells of the image |

- The four channels are not masked, so each crop also shows neighboring cells. `_mask.png` marks
  the whole center cell, not just its nucleus.
- Each data set, and each KOLF2.1J well, is scaled separately: overall brightness is similar, but
  the same 8-bit value does not mean the same amount of protein between data sets, wells or HPA
  images. Within a KOLF2.1J well, brightness differences between cells are kept.
- HPA crops are slightly smoothed by the rescaling and can have zero padding at an edge; KOLF2.1J
  crops have no padding.

The scripts are kept simple so that each step can be followed in the code, rather than optimized
for speed. They process one image at a time and keep their settings as constants at the top of the
file.

## Contents

```
hpa_pipeline.py          HPA: FOV TIFFs -> segmentation -> crops -> QC
kolf_pipeline.py         KOLF2.1J: z-plane TIFFs -> background subtraction -> segmentation -> crops -> QC
qc.py                    QC rules of both data sets, used by both pipelines
requirements.txt         Python packages for both pipelines
LICENSE                  MIT, for the code
LICENSE-DATA             CC BY 4.0, for the example images and outputs
examples/
  hpa/input/             2 FOVs, 4 TIFFs each
  hpa/output/            what hpa_pipeline.py writes for them
  kolf/input/            2 z-planes, 4 TIFFs each, and upper_limits.csv
  kolf/output/           what kolf_pipeline.py writes for them
```

Each `output/` folder holds:
- `segmentation/`: per image, `<image>_cellmask.png`, the labeled cell mask (pixel value = cell
  number, so it looks black in most viewers; open it with a label color map, e.g. in Fiji or
  napari), and `<image>_overview.png`, a half-size picture of the segmentation (nucleus blue,
  microtubules red): green outline = cropped and passed QC, magenta = removed by QC, gray = not
  cropped. Cropped cells show their number at the crop center;
- `crops/`: the cells that pass QC;
- `removed/`: the cells removed by QC (all KOLF2.1J example cells pass, so your
  `output/kolf/removed/` stays empty and `examples/kolf/output` has no `removed/` folder);
- `qc.csv`: one row per cropped cell, with the QC measurements and the result of each rule.

## Install and run

Both pipelines segment with the HPA Cell Segmentator, in the packaging of
[HPACellSegmentatorPortable](https://github.com/CellProfiling/HPACellSegmentatorPortable).
Clone it into this folder, and put its two model files in its `models/` folder:

```bash
git clone https://github.com/cm4ai-anvil-dream-challenge/cm4ai-anvil-dream-preprocessing
cd cm4ai-anvil-dream-preprocessing
git clone https://github.com/CellProfiling/HPACellSegmentatorPortable
mkdir HPACellSegmentatorPortable/models
curl -L -o HPACellSegmentatorPortable/models/dpn_unet_nuclei_v1.pth https://zenodo.org/records/4665863/files/dpn_unet_nuclei_v1.pth
curl -L -o HPACellSegmentatorPortable/models/dpn_unet_cell_3ch_v1.pth https://zenodo.org/records/4665863/files/dpn_unet_cell_3ch_v1.pth
python3.12 -m venv venv     # the pinned packages need Python 3.11 or 3.12
# without a GPU, adding --extra-index-url https://download.pytorch.org/whl/cpu gives a much smaller install
venv/bin/pip install -r requirements.txt
venv/bin/python hpa_pipeline.py
venv/bin/python kolf_pipeline.py
```

- Run the scripts from the repository root.
- They write to `output/hpa/` and `output/kolf/` (not tracked by git). Compare these with
  `examples/hpa/output/` and `examples/kolf/output/`.
- Both run on a CPU (`DEVICE = "cpu"`): about a minute each for the two example images on 4 cores.
  Set `DEVICE = "cuda"` to use a GPU. The example outputs were made on a CPU; on a GPU the masks can
  differ slightly, which can move a crop center by a few pixels.
- The segmentator prints `please compile abn` and some `SourceChangeWarning`s when it loads its
  models. Both are harmless.
- Cell numbers (`cell<N>`) are segmentation labels and can differ between machines; match cells by
  their crop center in `qc.csv`.

## HPA pipeline

### Input

Each FOV is four TIFFs: `<fov>_blue.tif` (nucleus), `<fov>_green.tif` (protein of interest),
`<fov>_red.tif` (microtubules) and `<fov>_yellow.tif` (endoplasmic reticulum). The pixel size is
read from the TIFF resolution tags. Most HPA FOVs are 2048×2048 and 16-bit, at 0.0800885 µm per
pixel, but bit depth, size and pixel size vary between FOVs. The pipeline reads them from each file.

### Steps

**1. Segmentation.** The HPA Cell Segmentator (models `dpn_unet_nuclei_v1` and
`dpn_unet_cell_3ch_v1`) runs at 1/4 scale. Nuclei are segmented on the nucleus channel, and cells on
the microtubule, ER and nucleus channels. Each cell gets the number of its nucleus. The protein
channel is not used.

**2. Cell filter and 1024×1024 crops.** With `FILTER_CELLS = True`, three kinds of cells are left
out:
- cells whose nucleus is within 4 px of the FOV border;
- cells smaller than 2,500 px;
- cells whose nucleus is smaller than 1,000 px.

For each remaining cell, a 1024×1024 crop is cut around the center of the cell's bounding box. The
four channels are not masked and keep the source bit depth. Where the crop runs past the edge of
the FOV, it is filled with 0. The cell mask is the cell's label, dilated with a 7×7 gray dilation,
so it reaches 3 px beyond the segmented outline.

**3. Rescaling and 512×512 crops.** Each 1024 crop is rescaled to 0.0941867 µm per pixel (the
KOLF2.1J pixel size; usually a factor 0.0800885 / 0.0941867 ≈ 0.850) with area interpolation, and
the central 512×512 window is kept, so a crop covers 48.2 µm. A 16-bit value `v` becomes
`round(v × 255 / 65535)`; 8-bit values are kept. There is no other intensity normalization.

**4. Quality control.** See [Quality control](#quality-control) (`qc.py`).

## KOLF2.1J pipeline

### Input

Each image is one z-plane of a confocal z-stack, as four 2160×2160 16-bit TIFFs:
`<image>_blue.tif` (nucleus), `<image>_green.tif` (protein of interest), `<image>_red.tif`
(microtubules) and `<image>_yellow.tif` (endoplasmic reticulum). The pixel size is 0.0941867 µm.

`upper_limits.csv` gives, for each image and channel, the intensity that becomes 255 in the 8-bit
images (`upper`, step 4) and the limit used for QC (`upper_qc`, step 5). Two things were computed
before this pipeline: the z-plane, which needs the whole z-stack (not included), and the upper
limits, which need all FOVs of a well:

- **The z-plane.** Each stack is 20 or 25 planes deep. The plane used is the sharpest one on the
  nucleus channel. Sharpness is the median of the standard deviations of the four quadrants of the
  plane.
- **The upper limits.** They are computed per well, from every pixel of the chosen planes of all
  FOVs of the well, after background subtraction (step 1):

  | channel | `upper` | `upper_qc` |
  |---|---|---|
  | nucleus | max(99.5th percentile, 100) / 0.8 | max(99.5th percentile, 100) |
  | protein | max(99th percentile, 100) / 0.6 | max(99th percentile, 100) |
  | microtubules, ER | max(99.5th percentile, 100) | the same |

  The floor of 100 keeps a well with little or no specific staining from having its detector noise
  stretched over the full range. The factors 0.8 and 0.6 bring the nucleus and protein channels to
  the brightness of the HPA crops. The QC thresholds were set at the brightness without these
  factors, so QC uses `upper_qc`. The file lists the limits per image, but all images of a well
  have the same values, so differences between cells of a well are kept.

### Steps

**1. Background subtraction.** A rolling ball of radius 500 px (47 µm; a cell is about 20 µm
across) is subtracted from each channel. For speed, the background is estimated on a copy shrunk
8×.

**2. Segmentation.** The HPA Cell Segmentator segments nuclei and cells as in the HPA pipeline, on
the raw plane, with each channel first scaled linearly to HPA brightness (for segmentation only).
The KOLF2.1J challenge crops were segmented with a consensus of the HPA Cell Segmentator,
Cellpose-SAM and Cellpose-DINO, so their masks differ from the masks made here. The HPA
Cell Segmentator is used here because the HPA pipeline needs it anyway, but it is not the best tool
for every image: in `kolf_example_2` it merges two pairs of touching cells (`cell35`, `cell40`;
both pass QC). Newer models such as Cellpose-SAM and Cellpose-DINO
([cellpose](https://github.com/MouseLand/cellpose), models `cpsam_v2` and `cpdino`) may segment
your images better, alone or combined.

**3. Crop windows.** Each cell gets one 512×512 window, centered on the center of the cell's
bounding box. Cells whose window would run past the edge of the image are left out. This also
leaves out whole cells close to the edge (gray in the overview), whereas HPA crops at the FOV edge
are zero-padded.

**4. 8-bit.** Each channel is mapped linearly: 0 stays 0, `upper` becomes 255, and brighter values
are clipped. There is no lower limit, because background subtraction already sets many pixels to
exactly 0. The mask is made as in the HPA pipeline: the cell's label, dilated with a 7×7 gray
dilation, so it reaches 3 px beyond the segmented outline. The challenge masks also kept only the
connected part of the mask under the crop center, so a few of them differ slightly.

**5. Quality control.** See [Quality control](#quality-control) (`qc.py`).

## Quality control

`qc.py` holds example QC rules for both data sets, based on the QC of the challenge data
(differences are noted). A cell is removed if any rule of its data set applies. The two rule sets
differ because the data differ:

- **HPA** (`hpa_qc`) looks only at the five 512×512 PNGs of each cell. Cells at the FOV border or
  too small are left out before cropping (step 2). HPA crops can run past the FOV, hence `padded`.
  Some cells are larger than the crop, so their mask covers the whole crop and cannot show whether
  the cell has a nucleus. For these cells, QC counts the nuclei in the crop instead.
- **KOLF2.1J** (`kolf_qc`) also uses the full-size masks (geometry rules) and compares each cell
  with the other cells of its image. The image rules use the 8-bit images made with `upper_qc`.
  KOLF2.1J crops never run past the image, so there is no `padded` rule.

Three rules are shared: `nucleus_free_mask`, `no_nucleus_in_crop` and `faint_reference`.

### HPA

| rule | the cell is removed if |
|---|---|
| `nucleus_free_mask` | the mean of the nucleus channel inside the mask is below 0.2 × its mean outside the mask (outside = pixels outside the mask that are not 0 in all four channels, which excludes padding and empty background). The mask then holds no nucleus, for example a cell segmented from cytoplasm only |
| `padded` | more than 50% of the crop is padding: rows and columns at the crop edge that are 0 in all four channels, because the crop ran past the edge of the FOV |
| `no_nucleus_in_crop` | the nucleus channel is all 0 or its 99th percentile is ≤ 2; or the mask covers the whole crop and no nucleus is found in the crop |
| `faint_reference` | the microtubule or ER channel is all 0 or its 99th percentile is ≤ 2 |
| `broken_mask` | the mask is empty; or the mask covers the whole crop and the crop holds two or more large nuclei (≥ 8,000 px) |

For a whole-crop mask, the nuclei are counted on the nucleus channel: Gaussian smoothing (σ = 2),
an Otsu threshold (at least 10), opening and hole filling; objects of at least 3,000 px count as
nuclei.

### KOLF2.1J

| rule | the cell is removed if |
|---|---|
| `small` | the cell is smaller than 2,500 px or its nucleus smaller than 1,000 px |
| `fragmented` | the cell's largest connected part holds less than 95% of it |
| `no_nucleus` | the cell has no nucleus, or less than 90% of its nucleus lies inside it |
| `nucleus_free_mask` | in the crop, the mean of the nucleus channel inside the mask is below 0.2 × its mean outside |
| `nucleus_dim` | the mean nucleus signal of its nucleus is below 0.2 × the median over all cells of the image |
| `no_marker` | the mean microtubule + ER signal of its cytoplasm is below 0.1 × the median over all cells of the image |
| `no_nucleus_in_crop` | the nucleus channel of the crop is all 0 or its 99th percentile is ≤ 2 |
| `faint_reference` | the microtubule or ER channel of the crop is all 0 or its 99th percentile is ≤ 2 |
| `nucleus_faint` | the mean nucleus signal of its nucleus is below 0.3 × the median over the cropped cells of the image that pass the geometry rules (the challenge data used all images of a well) |
| `nucleus_small` | its nucleus covers less than 2,000 px of the crop window |

`nucleus_faint` and `nucleus_small` are only evaluated for cells that pass the geometry rules
(`small`, `fragmented`, `no_nucleus`); for the other cells `qc.csv` records False. `qc.csv` also
records `touches_border` (the cell touches the edge of the image), which is not a rule. For the
challenge data, some whole FOVs were also removed, which needs the whole data set.

The HPA Cell Segmentator already removes small cells and nuclei and keeps nuclei inside cells, so
`small`, `no_nucleus` and the size filter of HPA step 2 rarely apply to its masks; they matter
mainly with other segmentation tools.

## Examples

| example | data set | cell line | antibody | cropped cells | pass QC |
|---|---|---|---|---|---|
| `hpa_example_1` | HPA | U-251MG | HPA023447 | 9 | 8 |
| `hpa_example_2` | HPA | U2OS | HPA052434 | 8 | 7 |
| `kolf_example_1` | KOLF2.1J | KOLF2.1J | HPA048321 | 25 | 25 |
| `kolf_example_2` | KOLF2.1J | KOLF2.1J | HPA060287 | 32 | 32 |

In the HPA examples, one removed cell fails `faint_reference` (example 1) and one fails
`nucleus_free_mask` (example 2: a large cell whose crop, centered on its bounding box, misses its
nucleus). The two KOLF2.1J example images are brighter than their wells, so
their microtubule and ER crops are more often saturated (255) than most challenge crops.

### Comparison with the challenge data

The example outputs were made on a CPU with the package versions in `requirements.txt`. On images
of the challenge data, starting from the same segmentation, the pipelines gave the same crops and QC
decisions as the challenge data; a few masks differ slightly at the outline.

## Adapting to other data

The settings are constants at the top of each script. For other images, check at least:

- **Pixel size and crop size.** Crops are 512×512 at 0.0941867 µm per pixel. The pixel areas in
  the cell filter and the QC rules were set at the pixel sizes of these images.
- **Channels.** Both pipelines expect nucleus, protein, microtubule and ER channels, as
  `_blue/_green/_red/_yellow.tif`.
- **Intensity scaling.** HPA images are mapped over the full 16-bit range; KOLF2.1J images use
  limits computed per well (`upper_limits.csv`). Choose limits that bring your images to an
  intensity distribution similar to that of these two data sets, for example similar per-channel
  percentiles.
- **Segmentation.** The HPA Cell Segmentator can merge touching cells; other models may suit your
  images better (KOLF2.1J step 2).
- **QC thresholds.** They were chosen for these two data sets. Check what each rule removes
  on your data before using it.

To process a full data set, optimize the scripts for speed first, or it will take a long time (the
two example images already take about a minute). For example:

- segment on a GPU (`DEVICE = "cuda"`) and pass several images per call, since the segmentator
  takes lists of images;
- process images in parallel, for example one job per plate or well on a cluster;
- dilate each cell's mask in a window slightly larger than its crop, not in the whole image
  (`crop_mask` in `kolf_pipeline.py`, and the mask step in `hpa_pipeline.py`);
- save intermediate results, such as background-subtracted images and segmentation masks, so that
  later steps can be rerun without recomputing them.

## Data

- **HPA:** the example images are from the Human Protein Atlas, v25.proteinatlas.org. Please cite
  Thul PJ et al. A subcellular map of the human proteome. Science 356, eaal3321 (2017).
- **KOLF2.1J:** the example images were generated by the Cell Maps for Artificial Intelligence
  (CM4AI) project. Please cite Clark T et al. Cell Maps for Artificial Intelligence: AI-Ready Maps
  of Human Cell Architecture from Disease-Relevant Cell Lines. bioRxiv 2024.05.21.589311 (2024),
  https://doi.org/10.1101/2024.05.21.589311. Both antibodies show no specific staining in KOLF2.1J
  and are not part of the challenge data.

All TIFFs hold the original pixel values and resolution tags; their other metadata was removed.
Changes made in the example outputs: the images were cropped to single cells, rescaled (HPA) or
background-subtracted (KOLF2.1J), and converted to 8 bit.

## License

The code is licensed under the MIT license (`LICENSE`). The example images and outputs are licensed
under CC BY 4.0 (`LICENSE-DATA`), like the challenge data. When you use the images, give the
attribution in Data.
