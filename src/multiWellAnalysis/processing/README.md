# phenotypr - BF processing pipeline

A modular Python pipeline for preprocessing, segmentation, and quantitative
analysis of high-throughput brightfield time-lapse microscopy data, with a focus
on bacterial biofilm growth and morphology in multi-well plate experiments.

This repository provides the **core analysis library** used to:
- preprocess raw brightfield images
- perform registration and drift correction
- compute biofilm masks
- extract biomass and optical density (OD) time series
- support downstream feature extraction and visualization

The codebase is designed to be **reproducible, scriptable, and scalable** for
large plate-based experiments.

---

## Scope and philosophy

This repository intentionally contains **only reusable analysis code**.

It does **not** contain:
- raw microscopy data
- processed images or outputs
- experimental results
- plate layouts or assay-specific metadata

Experiment-specific scripts that *use* this library (e.g. plate runners,
parameter sweeps) are maintained separately.

---

## Core functionality

The main analysis stages are:

1. **Preprocessing**
   - Local contrast normalization
   - Optional downsampling
   - Gaussian smoothing

2. **Registration**
   - Phase-correlation-based frame alignment
   - Drift correction using normalized reference images

3. **Segmentation**
   - Binary biofilm mask computation
   - Optional dust / artifact correction

4. **Quantification**
   - Biomass time series from masked images
   - Optical density (OD) estimation when reference images are provided

5. **Visualization**
   - Overlay generation for quality control
   - Biomass curves and peak-frame summaries

---

## Repository structure

```text
multiWellAnalysis/ \
├── multiWellAnalysis/ \
│ ├── analysis_main.py # main timelapse processing logic \
│ ├── preprocessing.py \
│ ├── registration.py \
│ ├── segmentation.py \
│ ├── feature_extraction.py \
│ ├── plotting_tools.py \
│ ├── io_utils.py \
│ └── helpers.py \
├── notebooks/ # exploratory notebooks (ignored by default) \
├── .gitignore \
└── README.md
```


---

## Typical usage

This package is intended to be imported and driven by **external scripts**, for
example:

- plate-level batch runners
- high-throughput screening pipelines
- experiment-specific workflows

A minimal usage pattern looks like:

```python
from multiWellAnalysis.analysis_main import timelapse_processing

masks, biomass, od = timelapse_processing(
    images=stack,
    block_diameter=101,
    ntimepoints=ntimepoints,
    shift_thresh=50,
    fixed_thresh=0.012,
    dust_correction=True,
    outdir=outdir,
    filename=well_id,
    image_records=[]
)
```

---

## Versioning

This repository uses semantic versioning.
- v1.0: frozen, stable implementation of the biofilm masking and biomass extraction pipeline used in current analyses.
Future versions may extend feature extraction or add alternative segmentation strategies, but v1.0 behavior is intended to remain reproducible.

## Authors
Seh Na Mellick, Jojo Prentice, Andrew Bridges
CMU Ray and Stephanie Lane Computational Biology Department
CMU Department of Biological Sciences

## License 
This repository is private and currently intended for academic use. A formal open-source license may be added in a future release.


