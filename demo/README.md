# Demo dataset

A small, real dataset for trying biofilm-processing (µPULLI-I) and µPULLI-DL without
downloading anything. It runs in under a minute on a normal computer.

## What it is

Four wells from one *Vibrio cholerae* transposon-mutant reimaging plate, imaged on a
Cytation 5 with a 10× objective (0.697 µm/pixel), brightfield, once per hour for 30 hours.
The plate is part of the published dataset (BioImage Archive
[S-BIAD3830](https://doi.org/10.6019/S-BIAD3830); source plate
`Plate2_Drawer3 02-Oct-2025 11-50-13`).

To keep the download small, the images were reduced in two ways:

- **Cropped** to a 512 × 512 pixel window (357 × 357 µm) of the original 1992 × 1992 field,
  at the same position in every well (rows 96–607, columns 1440–1951). That window was
  chosen because its biomass curves best matched the full-field curves for these four wells.
- **Every other frame** was kept: 16 frames at 0, 2, 4 … 30 h. They are renumbered
  `001`–`016`, so frame *n* was taken at (*n* − 1) × 2 hours.

Pixel values are the original 16-bit camera counts, unchanged. Files are stored with
lossless zlib compression. Each file keeps the Cytation XML header that the pipeline reads
for objective and pixel size; instrument serial numbers and acquisition-PC file paths in
that header were replaced with `REDACTED`.

```
demo/
├── raw/demo_plate/          64 TIFFs: <well>_02_1_1_Bright Field_<001–016>.tif
├── plate_layout.csv         which strain is in which well
└── expected/biomass.csv     biomass curves you should get back (see below)
```

| Well | Strain | Phenotype in the genome-wide screen |
|---|---|---|
| F10 | wild type | reference |
| D11 | *flaA*::Tn | High Biofilm |
| E3 | *pdhE2*::Tn | High Biofilm |
| C6 | *manB*::Tn | Dispersal Defect |

## Running it

See **Demo** in the [main README](../README.md#demo).

## What the results should show

`expected/biomass.csv` holds the biomass curves from the reference run (parameters as in
the main README). Your numbers should match to within rounding if you use the pinned
environment (`environment.yml`). With a newer environment, small differences in the last
digit are possible.

- **Wild type (F10)** forms a small biofilm that peaks around 10–12 h (biomass ≈ 0.005)
  and has dispersed completely by 14 h.
- **flaA (D11)** and **pdhE2 (E3)** form much larger biofilms, peaking about 6–9× higher
  than wild type (0.029 at 22 h and 0.043 at 24 h).
- **manB (C6)** never forms a large biofilm, but unlike wild type it does not fully
  disperse: biomass is still ≈ 0.005 at 30 h, when wild type is at zero.

**Wild type has no colony-level output** (`F10_02_perColonyFeatures.csv` is not written).
This is expected for this crop. Colony tracking starts from the first of two consecutive
frames with biomass ≥ 0.005. In this window, wild type passes that threshold in only one
of the sampled frames, so tracking falls back to a seed frame with no colonies large
enough to label. The full-size, hourly images of the same well do produce colony features.
