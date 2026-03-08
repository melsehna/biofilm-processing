import numpy as np
import pandas as pd
import os

baseDir = '/mnt/data/trainingData/241010_105227_Plate_1/processedImages'
plateDir = 'mnt/data/trainingData/241010_105227_Plate_1'
npz = np.load(os.path.join(baseDir, 'A5_trackedLabels_allFrames_trackingVec_v2.npz'))
labels = npz['labels']
tracked = npz['trackedFrames']
biomassOK = npz['biomassValidFrames']

assert np.all((labels.sum(axis=(0,1)) > 0) == (tracked & biomassOK))


border = np.unique(npz['borderLabels'])
assert not np.any(np.isin(labels, border))

df = pd.read_csv('/mnt/data/trainingData/241010_105227_Plate_1/A5_colonyFeatures_colFeats_trackingVec_v2.csv')
assert set(df['frame']).issubset(set(npz['frames']))

for t in range(labels.shape[2]):
    if labels[:,:,t].max() == 0:
        assert t not in set(df['frame'])
