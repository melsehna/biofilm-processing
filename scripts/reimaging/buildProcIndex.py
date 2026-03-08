#!/usr/bin/env python3

from pathlib import Path
import re
from glob import glob
import pandas as pd


RAW_BASE = Path(
    '/mnt/bridgeslab/Good imaging data/'
    'TN-Library_imaging/10x_data/Results/Final_Re-Imaging'
)

ANNOT_XLSX = Path(
    '/mnt/data/reimaging/Reimaging_plates_annotated.xlsx'
)

OUT_BASE = Path('/mnt/data/reimaging/index')
OUT_BASE.mkdir(parents=True, exist_ok=True)

OUT_INDEX = OUT_BASE / 'reimaging_index.csv'


rows = []

plate_dir_pattern = re.compile(r'Plate(\d+)_')

for plate_dir in RAW_BASE.iterdir():
    if not plate_dir.is_dir():
        continue

    m = plate_dir_pattern.match(plate_dir.name)
    if not m:
        continue

    repPlate = int(m.group(1))

    # Discover wells from Bright Field TIFFs
    tif_files = glob(str(plate_dir / '*.tif'))

    by_well = {}
    for f in tif_files:
        name = Path(f).name
        well = name.split('_', 1)[0]

        # Defensive: valid 96-well IDs only
        if not re.match(r'^[A-H][0-9]{1,2}$', well):
            continue

        by_well.setdefault(well, []).append(f)

    for repWell in sorted(by_well):
        rows.append({
            'repPlate': repPlate,
            'repWell': repWell,
            'plateDir': plate_dir.name,
            'platePath': str(plate_dir),
            'imagingRun': plate_dir.name
        })


if not rows:
    raise RuntimeError('No re-imaging data discovered on disk')

fs_index = pd.DataFrame(rows)



annot = pd.read_excel(ANNOT_XLSX)

# Explicitly rename to avoid ambiguity
annot = annot.rename(columns={
    'Plate': 'srcPlate',
    'Well': 'srcWell',
    'Rep Plate': 'repPlate',
    'Well.1': 'repWell',
    'Gene.Locus': 'geneLocus',
    'Name': 'geneName',
    'Function': 'function'
})

required = {
    'srcPlate', 'srcWell',
    'repPlate', 'repWell',
    'geneLocus', 'geneName', 'function'
}
missing = required - set(annot.columns)
if missing:
    raise RuntimeError(f'Missing required columns in annotation: {missing}')

# Clean repPlate
annot['repPlate'] = pd.to_numeric(annot['repPlate'], errors='coerce')
annot = annot.dropna(subset=['repPlate'])
annot['repPlate'] = annot['repPlate'].astype(int)

# Normalize wells
annot['repWell'] = annot['repWell'].astype(str).str.strip()
annot['srcWell'] = annot['srcWell'].astype(str).str.strip()

annot = annot[[
    'repPlate',
    'repWell',
    'geneLocus',
    'geneName',
    'function',
    'srcPlate',
    'srcWell'
]]


index = fs_index.merge(
    annot,
    on=['repPlate', 'repWell'],
    how='left',
    validate='many_to_one'
)


n_missing = index['geneLocus'].isna().sum()
if n_missing > 0:
    print(f'WARNING: {n_missing} wells missing annotation')

# Ensure stable ordering
index = index.sort_values(
    ['repPlate', 'plateDir', 'repWell']
).reset_index(drop=True)


tmp = OUT_INDEX.with_suffix('.tmp')
index.to_csv(tmp, index=False)
tmp.replace(OUT_INDEX)

print(f'Wrote {len(index)} rows')
print(f'Annotated index: {OUT_INDEX}')
