#!/usr/bin/env python3
"""
Read per-image acquisition metadata from Cytation5/BioSpa TIFF files.

Cytation5 stores acquisition parameters as XML in the ImageDescription TIFF tag.
This module extracts objective magnification and pixel→micron conversion from that XML.
"""

from xml.etree import ElementTree as ET


def readCytationMeta(tifPath):
    """Read objective magnification and px→μm from a Cytation5 TIFF.

    Reads the BTIImageMetaData XML stored in the ImageDescription tag.
    Only opens the first page; does not load pixel data.

    Parameters
    ----------
    tifPath : str
        Path to any single-frame Cytation5 TIFF.

    Returns
    -------
    dict with:
        'objective' : int   — objective magnification (e.g. 4, 10, 20, 40)
        'pxToUm'   : float  — microns per pixel (e.g. 0.697 for 10x)

    Raises
    ------
    ValueError  if the expected XML fields are missing or can't be parsed.
    """
    import tifffile

    with tifffile.TiffFile(tifPath) as tif:
        page = tif.pages[0]
        if 270 not in page.tags:
            raise ValueError(f'No ImageDescription tag in {tifPath}')
        xmlStr = page.tags[270].value

    root = ET.fromstring(xmlStr)
    acq = root.find('ImageAcquisition')
    if acq is None:
        raise ValueError(f'No <ImageAcquisition> element in metadata of {tifPath}')

    def _get(elem, tag):
        node = elem.find(tag)
        if node is None:
            raise ValueError(f'Missing <{tag}> in ImageAcquisition metadata of {tifPath}')
        return node.text.strip()

    objective = int(_get(acq, 'ObjectiveSize'))
    pixelWidth = int(_get(acq, 'PixelWidth'))
    imageWidthMicrons = float(_get(acq, 'ImageWidthMicrons'))

    if pixelWidth == 0:
        raise ValueError(f'PixelWidth is 0 in {tifPath}')

    pxToUm = imageWidthMicrons / pixelWidth

    return {'objective': objective, 'pxToUm': pxToUm}


def probeSuffixMeta(wells, logFn=None):
    """Probe one TIFF per magnification suffix to read acquisition metadata.

    Parameters
    ----------
    wells : dict[str, list[str]]
        Well → list of TIFF paths, as returned by discoverWells().
        Well keys have the form 'A1_03', 'B2_03', etc.
    logFn : callable, optional

    Returns
    -------
    dict[str, dict]  suffix → {'objective': int, 'pxToUm': float}
        e.g. {'_03': {'objective': 10, 'pxToUm': 0.6973}}
        Missing/unreadable suffixes are omitted.
    """
    import re

    def log(msg):
        if logFn:
            logFn(msg)

    # Group one representative file per suffix
    suffixFile = {}
    for wellId, files in wells.items():
        m = re.search(r'(_\d+)$', wellId)
        suffix = m.group(1) if m else ''
        if suffix not in suffixFile and files:
            suffixFile[suffix] = files[0]

    result = {}
    for suffix, tifPath in sorted(suffixFile.items()):
        try:
            meta = readCytationMeta(tifPath)
            result[suffix] = meta
            label = suffix if suffix else '(no suffix)'
            log(f'  Metadata {label}: {meta["objective"]}x objective, '
                f'{meta["pxToUm"]:.4f} μm/px  [{tifPath}]')
        except Exception as e:
            log(f'  Metadata {suffix}: could not read ({e})')

    return result
