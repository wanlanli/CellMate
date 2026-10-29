# CellMate Quickstart

Three annotated, runnable demos of the core CellMate pipeline, distilled
from the informal, real-usage scripts in [`../tmp`](../tmp). Each step has
both a notebook (with explanations, visualizations, and sanity checks) and
a command-line script (for batch-processing many files without a notebook).

| step | notebook | script | what it does |
|---|---|---|---|
| 1 | [`01_rigid_registration.ipynb`](01_rigid_registration.ipynb) | [`01_rigid_registration.py`](01_rigid_registration.py) | Removes stage drift from a raw time-lapse movie. |
| 2 | [`02_tracking_movie.ipynb`](02_tracking_movie.ipynb) | [`02_tracking_movie.py`](02_tracking_movie.py) | Links segmented cells across frames; detects division/fusion (mating) events. |
| 3 | [`03_measure_image.ipynb`](03_measure_image.ipynb) | [`03_measure_image.py`](03_measure_image.py) | Extracts per-cell morphology (area, axis lengths, skeleton length, ...) and spatial relationships (neighbors, distance, angle) from a labeled mask. |

These steps mirror a typical CellMate pipeline: register &rarr; segment
(not covered here -- CellMate has no segmentation code of its own; segment
the registered DIC/brightfield channel into a per-cell instance mask with
whatever model/tool is on hand, and hand-correct any mistakes before
tracking) &rarr; track &rarr; measure. Steps 2 and 3 use one of the bundled
example movies that has already been registered and segmented, so you can
run either one independently.

## Setup

These demos need the same environment used to develop CellMate: Python 3
with `numpy`, `pandas`, `matplotlib`, `networkx`, `scikit-image`,
`SimpleITK`, and `tqdm` installed, plus `cellmate` importable (either
installed, or simply run from/next to the repository root -- every
notebook/script here locates the repo root automatically by walking up from
its own location).

## Running the notebooks

```bash
cd demo/quickstart
jupyter lab   # or: jupyter notebook
```

Each notebook can be run top-to-bottom (`Run All`) on the bundled example
data with no arguments or configuration.

## Running the command-line scripts

Each script works out of the box on the bundled example data:

```bash
python 01_rigid_registration.py
python 02_tracking_movie.py
python 03_measure_image.py
```

...and accepts arguments for pointing at your own data, e.g.:

```bash
python 01_rigid_registration.py --input my_movie.tif --output my_movie_rigid.tif
python 02_tracking_movie.py --input my_movie_rigid_segmented.tif --mask-channel -1
python 03_measure_image.py --batch --input my_movie_tracked.tif --output measurements.csv
```

Run any script with `--help` for the full list of options. By default,
outputs are written to an `outputs/` folder created next to the scripts
(not checked into version control).

## Where the example data comes from

`../yav2104yav2105_mating_example_01.tif` and
`../YAV2104xYAV2105_20220826_exp011_05_R3D_rigid_003002.tif` are existing
example movies bundled in `../`. Nothing in this folder was added anywhere
else in the repository -- these demos only read from `../` and write into
their own local `outputs/`.
