# \# AE-LRHMA: Autoencoder-Enhanced Latent Representation Hierarchical Mondrian Anonymization

# 

# This repository contains the reference implementation of \*\*AE-LRHMA (Autoencoder-Enhanced Latent Representation Hierarchical Mondrian Anonymization)\*\*, an anonymization method for structured/tabular data under the k-anonymity framework with additional sensitive-distribution constraints.

# 

# AE-LRHMA first learns a compact \*\*latent representation\*\* of quasi-identifiers using an \*\*autoencoder\*\*, then performs hierarchical Mondrian-style partitioning in the latent space and applies a local (k,e)-MDAV-like microaggregation with constraints on sensitive attribute distribution. Finally, it generalizes records back in the original attribute space.

# 

# > Paper target venue: \*\*MDPI\*\* (Manuscript accompanying this codebase).  

# > Datasets: \*\*UCI Bank Marketing\*\* and \*\*UCI Adult Census Income\*\* (public).  

# > Please ensure the column naming in your local dataset matches the expected schema (see below).

# 

# ---

# 

# \## Repository Contents

# 

# Main method:

# \- `LR\_HMA.py` — AE-LRHMA main implementation (class `LRHMA`). It trains an autoencoder and runs hierarchical partition + local microaggregation.  

# &nbsp; (See the `\_\_main\_\_` section for an example run.)  

# 

# Baselines / variants:

# \- `mdav3.py` — Strict (k,e)-MDAV baseline implementation (`StrictKEMDAV`).

# \- `apmca-para.py` — APMCA serial implementation (`APMCAAlgorithm`).

# \- `apmca-para-xzw.py` — Parallel APMCA implementation (`APMCAAlgorithmParallel`) with multi-threading.

# 

# Visualization:

# \- `数据聚类图.py` — 3D visualization of AE latent space before anonymization and grouped results after AE-LRHMA (saves figures).

# 

# ---

# 

# \## Requirements

# 

# \- Python 3.9+ recommended

# \- Core packages:

# &nbsp; - `numpy`, `pandas`, `scikit-learn`, `matplotlib`

# &nbsp; - `torch` (PyTorch) for the autoencoder in AE-LRHMA

# 

# Install with:

# 

# ```bash

# pip install numpy pandas scikit-learn matplotlib torch openpyxl

# 

