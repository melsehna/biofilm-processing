"""Build index CSV for the transposon set linking original paths to processed outputs."""
import os
import re
import pandas as pd
from glob import glob

BASEDIR = '/mnt/bridgeslab/Good imaging data/TN-Library_imaging'
OUTDIR = '/mnt/data/transposonSet'

PLATES = [
    'TN-Plate01_4x_10x_20x_40x_Discontinuous_Drawer1 06-Nov-2024 14-57-44',
    'TN-Plate02_4x_10x_20x_40x_Discontinuous_Drawer2 25-Oct-2024 17-17-58',
    'TN-Plate03_4x_10x_20x_40x_Discontinuous_Drawer2 29-Oct-2024 14-46-08',
    'TN-Plate04_4x_10x_20x_40x_Discontinuous_Drawer3 29-Oct-2024 14-46-35',
    'TN-Plate07_4x_10x_20x_40x_Discontinuous_Drawer3 31-Oct-2024 13-36-05',
    'TN-Plate08_4x_10x_20x_40x_Discontinuous_Drawer1 04-Nov-2024 15-56-04',
    'TN-Plate09_4x_10x_20x_40x_Discontinuous_Drawer2 04-Nov-2024 15-56-34',
    'TN-Plate10_4x_10x_20x_40x_Discontinuous_Drawer3 04-Nov-2024 15-57-04',
    'TN-Plate11_4x_10x_20x_40x_Discontinuous_Drawer2 06-Nov-2024 14-58-16',
    'TN-Plate12_4x_10x_20x_40x_Discontinuous_Drawer3 06-Nov-2024 14-58-50',
    'TN-Plate13_4x_10x_20x_40x_Discontinuous_Drawer1 20-Nov-2024 16-25-11',
    'TN-Plate14_4x_10x_20x_40x_Discontinuous_Drawer2 20-Nov-2024 16-25-39',
    'TN-Plate15_4x_10x_20x_40x_Discontinuous_Drawer3 20-Nov-2024 16-47-34',
    'TN-Plate16_4x_10x_20x_40x_Discontinuous_Drawer1 11-Nov-2024 16-06-24',
    'TN-Plate17_4x_10x_20x_40x_Discontinuous_Drawer2 11-Nov-2024 16-06-55',
    'TN-Plate18_4x_10x_20x_40x_Discontinuous_Drawer3 11-Nov-2024 16-07-25',
    'TN-Plate19_4x_10x_20x_40x_Discontinuous_Drawer1 18-Nov-2024 11-46-38',
    'TN-Plate20_4x_10x_20x_40x_Discontinuous_Drawer2 18-Nov-2024 11-46-59',
    'TN-Plate21_4x_10x_20x_40x_Discontinuous_Drawer3 18-Nov-2024 12-47-49',
    'TN-Plate22_4x_10x_20x_40x_Discontinuous_Drawer1 12-Dec-2024 15-50-10',
    'TN-Plate23_4x_10x_20x_40x_Discontinuous_Drawer2 12-Dec-2024 15-50-38',
    'TN-Plate24_4x_10x_20x_40x_Discontinuous_Drawer2 19-Dec-2024 15-41-47',
    'TN-Plate25_4x_10x_20x_40x_Discontinuous_Drawer3 25-Nov-2024 15-53-17',
    'TN-Plate26_4x_10x_20x_40x_Discontinuous_Drawer4 25-Nov-2024 16-15-04',
    'TN-Plate27_4x_10x_20x_40x_Discontinuous_Drawer1 27-Nov-2024 13-22-12',
    'TN-Plate28_4x_10x_20x_40x_Discontinuous_Drawer2 27-Nov-2024 13-22-40',
    'TN-Plate29_4x_10x_20x_40x_Discontinuous_Drawer3 27-Nov-2024 13-23-07',
    'TN-Plate30_4x_10x_20x_40x_Discontinuous_Drawer1 04-Dec-2024 15-50-29',
    'TN-Plate31_4x_10x_20x_40x_Discontinuous_Drawer2 04-Dec-2024 15-50-59',
    'TN-Plate32_4x_10x_20x_40x_Discontinuous_Drawer3 04-Dec-2024 15-51-30',
    'TN-Plate33_4x_10x_20x_40x_Discontinuous_Drawer1 06-Dec-2024 16-04-05',
    'TN-Plate34_4x_10x_20x_40x_Discontinuous_Drawer2 06-Dec-2024 16-04-34',
]

records = []

for plate_outer_name in PLATES:
    plate_label = re.match(r'(TN-Plate\d+)', plate_outer_name).group(1)
    outer_dir = os.path.join(BASEDIR, plate_outer_name)

    # Find inner plate dir (e.g. "241106_150118_Plate 1")
    inner_dirs = [
        d for d in os.listdir(outer_dir)
        if os.path.isdir(os.path.join(outer_dir, d)) and 'Plate' in d
    ]
    if not inner_dirs:
        print(f'WARNING: no Plate subdir in {outer_dir}')
        continue

    inner_name = inner_dirs[0]
    original_plate_dir = os.path.join(outer_dir, inner_name)

    # Find matching output dir
    output_plate_dir = os.path.join(OUTDIR, inner_name)
    proc_dir = os.path.join(output_plate_dir, 'processedImages')

    if not os.path.isdir(output_plate_dir):
        print(f'WARNING: no output dir for {plate_label}: {output_plate_dir}')
        continue

    # Find all wells from timeseries CSVs
    timeseries_files = sorted(glob(os.path.join(output_plate_dir, '*_timeseries.csv')))

    for ts_path in timeseries_files:
        ts_name = os.path.basename(ts_path)
        # e.g. A1_03_timeseries.csv -> well_mag = A1_03
        well_mag = ts_name.replace('_timeseries.csv', '')

        processed_path = os.path.join(proc_dir, f'{well_mag}_processed.tif')
        raw_reg_path = os.path.join(proc_dir, f'{well_mag}_registered_raw.tif')
        mask_path = os.path.join(proc_dir, f'{well_mag}_masks.npz')
        overlay_path = os.path.join(proc_dir, f'{well_mag}_overlay.mp4')

        records.append({
            'plate_label': plate_label,
            'plate_outer_dir': outer_dir,
            'plate_inner_dir': inner_name,
            'original_images_dir': original_plate_dir,
            'well_mag': well_mag,
            'timeseries_csv': ts_path if os.path.exists(ts_path) else '',
            'processed_tif': processed_path if os.path.exists(processed_path) else '',
            'registered_raw_tif': raw_reg_path if os.path.exists(raw_reg_path) else '',
            'masks_npz': mask_path if os.path.exists(mask_path) else '',
            'overlay_mp4': overlay_path if os.path.exists(overlay_path) else '',
        })

df = pd.DataFrame(records)
out_path = os.path.join(OUTDIR, 'transposon_set_index.csv')
df.to_csv(out_path, index=False)

print(f'Index written to: {out_path}')
print(f'  {len(df)} wells across {df["plate_label"].nunique()} plates')
print(f'  Timeseries: {(df["timeseries_csv"] != "").sum()}')
print(f'  Processed:  {(df["processed_tif"] != "").sum()}')
print(f'  Raw reg:    {(df["registered_raw_tif"] != "").sum()}')
print(f'  Masks:      {(df["masks_npz"] != "").sum()}')
print(f'  Overlays:   {(df["overlay_mp4"] != "").sum()}')
