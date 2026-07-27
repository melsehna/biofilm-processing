#!/usr/bin/env python
"""Per-dataset biology resolvers + config for the BIA deposit.

Joins the paper's per-well biology tables (~/uPULLI-figures) to the processed
wells. A well is IN the figure-analyzed deposit iff it appears in its table
(deduped on plateId,wellId — the tables duplicate each well per UMAP metric).

Fields marked TBD come from the manuscript/user (strain backgrounds, DMSO %);
they slot in here without touching the image conversion.
"""
import csv, os, re

FIG = os.path.expanduser("~/uPULLI-figures")

def _norm(p):
    """Normalize plateId for joining (spaces/underscores differ across tables)."""
    return re.sub(r"[\s_]+", "", p).lower()


# ---------- study-level / shared config ----------
ACQUISITION = dict(Channel="Bright Field", Microscope="BioTek Cytation 5",
                   GrowthTemp="37 C", ImagingTemp="30 C",
                   SeedingDensity="~1e4-1e5 CFU/mL",
                   PlateType="Corning polystyrene 96-well")

# ---------- K. pneumoniae ----------
# Strain background is TBD (user). NV_### are the transposon-collection isolate IDs;
# drawer->isolate from run_kleb_062926.sh. Genotype is authoritative from fig5 table.
KLEB_STRAIN_BG = "KPPR1"
KLEB_MEDIUM = ("M9 + glucose, Chelex-100-chelated (1x M9 salts, 100 uM CaCl2, 1 mM MgSO4, "
               "0.4% glucose); overnight in LB; OT-2 dilution first two steps in PBS, last two in M9")
KLEB_ISOLATE = {   # plate basename -> (drawerN, isolateID)
    "250311_124651_Plate 1": ("Drawer1", "NV_058"),
    "250311_125358_Plate 1": ("Drawer2", "NV_059"),
    "250311_130104_Plate 1": ("Drawer3", "NV_064"),
    "250311_130813_Plate 1": ("Drawer4", "NV_065"),
    "250311_131518_Plate 1": ("Drawer5", "NV_066"),   # waaL — not in figure, excluded
    "250311_132226_Plate 1": ("Drawer6", "NV_070"),
}
def _kleb_muttype(g):
    if g == "WT":
        return "wild-type"
    if re.search(r"[A-Z]\d+[A-Z]$", g):     # e.g. WzcQ395K
        return "point mutation"
    return "transposon insertion"           # per BIA email framing (K. pneumoniae transposon mutants)

def kleb_membership():
    """(normPlate, well) -> {'Genotype':..} for figure-analyzed kleb wells."""
    p = os.path.join(FIG, "fig5/data/kleb_embeddingUmap_coords.csv")
    out = {}
    for r in csv.DictReader(open(p)):
        out[(_norm(r["plateId"]), r["wellId"])] = {"Genotype": r["mutant"]}
    return out

def kleb_row(platBasename, well, genotype):
    """Biology columns for one kleb well (merged into the file-list row)."""
    drawer, iso = KLEB_ISOLATE.get(platBasename, ("", ""))
    return dict(Strain="Klebsiella pneumoniae", StrainBackground=KLEB_STRAIN_BG,
                Genotype=genotype, Gene=("" if genotype == "WT" else genotype),
                MutationType=_kleb_muttype(genotype), IsolateID=iso, Drawer=drawer,
                Medium=KLEB_MEDIUM)

# Lean columns: only attributes that VARY within this list. Constants (Strain,
# StrainBackground, Objective, PixelSize, Frames/interval/duration, Channel,
# Microscope, Medium, temps, seeding, plate type, pipeline/version/params) are
# constant across kleb and live in the web-form biosample/specimen/acquisition.
KLEB_STUDY_COLUMNS = [
    "Files", "Plate", "Well", "Genotype", "Gene", "MutationType", "IsolateID", "Drawer",
    "OriginalPlate", "OriginalWell",
]
# Annotation lists: only the two mandatory columns; AnnotationType/Method/pixel
# meaning are constant per list and live in the web-form Annotation section.
ANNOTATION_COLUMNS = ["Files", "source_image"]

# =========================== V. cholerae ===========================
# One study component, 4 perturbation subdirs. Strain background TBD (user).
VC_STRAIN_BG = "C6706str2 (O1 El Tor biotype)"
VC_MEDIUM = ("M9 + glucose + casamino acids (1x M9 salts, 100 uM CaCl2, 2 mM MgSO4, "
             "0.5% glucose, 0.5% casamino acids); overnight in LB")
VC_STRAIN = "Vibrio cholerae"

# known clean deletions vs point mutants among the 8 training genotypes
VC_KNOWN_POINT = {"luxO_D47E", "vpvC_W240R"}    # residue-change alleles
def _vc_known_muttype(g):
    if g == "WT": return "wild-type"
    if g in VC_KNOWN_POINT: return "point mutation"
    return "clean deletion"

# compound treatments (cluster fig5): token "strain_compound"
VC_COMPOUNDS = {   # compound token -> (Name, Identifier/notes, Concentration)
    "DMSO":    ("DMSO", "vehicle control", "0.5% v/v"),
    "antiBio": ("MAC13772", "biotin-biosynthesis inhibitor; dissolved in 0.5% DMSO", "100 uM"),
    "biotin":  ("biotin", "dissolved in water", "100 uM"),
    "nspd":    ("norspermidine", "dissolved in water", "100 uM"),
}
def _vc_compound_split(tok):
    """'WT_DMSO' -> (strain, compoundToken). 'bioD_biotin' -> ('bioD','biotin')."""
    i = tok.find("_")
    return (tok[:i], tok[i+1:]) if i >= 0 else (tok, "")

def _load(path, cols):
    out = {}
    for r in csv.DictReader(open(os.path.expanduser(path))):
        out[(_norm(r["plateId"]), r["wellId"])] = {c: r[c] for c in cols}
    return out

def vc_known_membership():
    return _load("~/uPULLI-figures/fig1/data/trainingEmbeddingUmap_coords.csv", ["mutant"])
def vc_transposon_membership():
    return _load("~/uPULLI-figures/fig3/data/reimagingUmap_nn10_md0.10_perGene_coords.csv",
                 ["mutant", "geneLocus", "function", "functionalGroup"])
def vc_cleandel_membership():
    return _load("~/uPULLI-figures/fig4/data/cleanDeletions_projectedCoords.csv", ["mutant"])
def vc_compound_membership():
    return _load("~/uPULLI-figures/fig5/data/compounds_projectedCoords.csv", ["mutant"])

def _blank_vc():
    return dict(Strain=VC_STRAIN, StrainBackground=VC_STRAIN_BG, Perturbation="",
               Genotype="", Gene="", GeneLocus="", Function="", FunctionalGroup="",
               MutationType="", Compound_Name="", Compound_Identifier="",
               Compound_Concentration="", Medium=VC_MEDIUM)

def vc_row(perturbation, bio):
    d = _blank_vc(); d["Perturbation"] = perturbation
    if perturbation == "knownMutant":
        g = bio["mutant"]; d.update(Genotype=g, Gene=("" if g == "WT" else g.split("_")[0]),
                                    MutationType=_vc_known_muttype(g)); d["_token"] = g
    elif perturbation == "transposon":
        g = bio["mutant"]; loc = bio.get("geneLocus", "")
        d.update(Genotype=g, Gene=g, GeneLocus=loc,
                 Function=bio.get("function", ""), FunctionalGroup=bio.get("functionalGroup", ""),
                 MutationType="transposon insertion")
        # filename token: "<name>_<locus>" when the gene has a real name, else just the locus
        named = g and loc and g.lower() != loc.lower() and not re.fullmatch(r"VC_?A?\d+", g)
        d["_token"] = f"{g}_{loc}" if named else (loc or g)
    elif perturbation == "cleanDeletion":
        g = bio["mutant"]; d.update(Genotype=g, Gene=("" if g == "WT" else g),
                                    MutationType=("wild-type" if g == "WT" else "clean deletion")); d["_token"] = g
    elif perturbation == "compound":
        strain, comp = _vc_compound_split(bio["mutant"])
        name, ident, conc = VC_COMPOUNDS.get(comp, (comp, "", ""))
        d.update(Genotype=strain, Gene=("" if strain == "WT" else strain),
                 MutationType=("wild-type" if strain == "WT" else "clean deletion"),
                 Compound_Name=name, Compound_Identifier=ident, Compound_Concentration=conc)
        d["_token"] = bio["mutant"]
    return d

# Lean: keep varying columns. GitCommit kept (2 values: training 11a1038 vs
# 709d014). Constants (Strain/StrainBackground/Objective/PixelSize/FrameInterval/
# Channel/Microscope/Medium/temps/seeding/plate/pipeline/params) -> web form.
VC_STUDY_COLUMNS = [
    "Files", "Plate", "Well", "Perturbation", "Genotype", "Gene", "GeneLocus",
    "Function", "FunctionalGroup", "MutationType",
    "Compound_Name", "Compound_Identifier", "Compound_Concentration",
    "Frames", "TotalDuration", "GitCommit", "OriginalPlate", "OriginalWell",
]

# =========================== Multispecies ===========================
MS_MEDIUM = "LB (grown overnight and imaged in LB, 100%)"
def ms_membership():
    return _load("~/uPULLI-figures/fig5/data/multispecies_100pctLB_10X_umap_coords.csv",
                 ["species", "LB_condition"])
def ms_speciesDir(species):
    return sanitize_species(species)
def sanitize_species(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")   # 'S. aureus' -> 'S_aureus'
def ms_row(bio):
    return dict(Species=bio["species"], LB_condition=bio.get("LB_condition", "100% LB"),
                Medium=MS_MEDIUM)
# Lean: Species varies (8); LB_condition/Objective/PixelSize/Frames/etc. are
# constant (all 100% LB, 10x, masks-only) -> web form.
MS_STUDY_COLUMNS = [
    "Files", "Plate", "Well", "Species", "OriginalPlate", "OriginalWell",
]
