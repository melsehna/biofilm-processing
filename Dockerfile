# Headless biofilm-processing pipeline (biofilm-processing-run / -test-well).
#
# This image reproduces the ANALYSIS, not the GUI. Qt GUIs in containers need X
# forwarding and are not worth it; the reproducible science is the headless CLI.
#
# Base: micromamba (conda-forge) so the installed numpy/scipy/scikit-image match
# the BLAS and versions the pipeline was validated on (see environment.yml).
# A PyPI-wheel base would install numerically different builds.
#
# Build (multi-arch: amd64 for HPC/x86, arm64 for Apple Silicon):
#   docker buildx build --platform linux/amd64,linux/arm64 \
#     -t ghcr.io/melsehna/biofilm-processing:0.5.0 --push .
#
# Run locally (bind-mount data in, outputs out):
#   docker run --rm -v /data:/data -v /out:/out \
#     ghcr.io/melsehna/biofilm-processing:0.5.0 \
#     biofilm-processing-run --plates /data/plateA -o /out --mag _03 --workers 8
#
# Run on an HPC cluster with Apptainer/Singularity (no root, no Docker daemon):
#   apptainer pull biofilm.sif docker://ghcr.io/melsehna/biofilm-processing:0.5.0
#   apptainer exec --bind /path/to/data,/path/to/output biofilm.sif \
#     biofilm-processing-run --plates /path/to/data/plateA -o /path/to/output --mag _03 --workers 40
FROM mambaorg/micromamba:1.5.10

# System libraries the Python lockfile cannot capture: OpenGL + glib for opencv,
# EGL/xkb for Qt offscreen rendering, and ffmpeg for the cv2.VideoWriter overlay
# fallback. Without these, `import cv2` / offscreen Qt fail on a clean node.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libegl1 \
        libxkbcommon0 \
        libdbus-1-3 \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

WORKDIR /app

# Solve the environment first (its own layer) so dependency installs are cached
# across code changes.
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /app/environment.yml
RUN micromamba install -y -n base -f /app/environment.yml && \
    micromamba clean --all --yes

# Install the package itself without re-resolving deps (already solved above).
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install --no-deps -e .

# Headless / offscreen defaults; single-thread BLAS so ProcessPoolExecutor workers
# don't oversubscribe (matches cli/run_pipeline.py).
ENV QT_QPA_PLATFORM=offscreen \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["biofilm-processing-run", "--help"]
