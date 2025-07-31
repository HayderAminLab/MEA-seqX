# **MEA-seqX: High‐resolution Profiling of Large‐scale Electrophysiological and Transcriptional Network Dynamics**

![Figure 1](https://github.com/HayderAminLab/MEA-seqX/assets/158823360/15b48459-642f-4311-837e-eb733005b5ba)

The platform integrates high-density microelectrode arrays, spatial transcriptomics, optical imaging, and advanced computational strategies for simultaneous recording and analysis of molecular and electrical network activities at the individual cell level. 

It enables the study of nested dynamics between transcription and function, revealing coordinated spatiotemporal dynamics in brain circuitry at both molecular and functional levels. 
This platform also allows for the identification of different cell types based on their distinct bimodal profiles and uses machine learning algorithms to predict network-wide electrophysiological features from spatial gene expression.

## **1. 📚 Contents**

  - data - additional data needed to use and test run the repository. NOTE: This data can only be combined with the provided Zenodo datasets and cannot be used for personal data.
  - help_functions - low-level functions used to clean and process nEphys data
  - README - current file
  - LICENSE - usage and redistrubution conditions
  - Multiscale Spatial Alignment.py - main top-level function used to overlay the SRT and nEphys data with automatic slice alignment
  - SRT Gene Expression.py - top-level functions used for preprocessing, filtering, normalizing, and plotting provided gene lists and select genes for further analysis
  - SRT_nEphys Network Activity Features.py - main top-level, multi-step functions used to calculate specified network activity features from nEphys data and correlate them with gene expression values from SRT gene lists
  - XGBoost Algorithm Prediction.py - top-level function used to predict nEphys network activity features from SRT gene list expression values using XGBoost Algorithm
  - SRT_nEphys Network Topological Metrics.py - main top-level, multi-step functions used to calculate specified network topological metrics from nEphys data and correlate them with gene expression values from SRT gene lists
  - Non-Negative Matrix Factorization.py - top level function used to identify distinct spatiotemporal patterns across the combined SRT and n‐Ephys modalities


## **2. 🚀 Getting Started**

  - Download SRT and nEphys data from Zenodo and separate into subfolders based on condition i.e. SD and ENR. Include data from both the repository and downloaded datasets.
  - If using personal data, format folder structure as instructed for the provided data. 
  - Clone the respository locally.
       > git clone https://github.com/HayderAminLab/MEA-seqX.git
  - Go to the directory and work through the codes in the following order:
    1. Multiscale Spatial Alignment
    2. SRT Gene Expression 
    3. SRT_nEphys Network Activity Features
    4. XGBoost Algorithm Prediction
    5. SRT_nEphys Network Topological Metrics
    6. Non-Negative Matrix Factorization
   NOTE: For multi-step functions follow the order specified in the code. 
     
## **3. 🧠 📦 Data**

The following datasets have been provided for using and test running the MEA-seqX platform. Additional data needed to run the repository for these datasets are included in the repository data subfolder.
  - Dataset of Spatial Transcriptomics Mouse Hippocampal Slices (https://doi.org/10.5281/zenodo.10626259)
  - Dataset of HD-MEA n-Ephys Mouse Hippocampal Slices (https://doi.org/10.5281/zenodo.10620559)

## **4. 🧩 ⚙️ Requirements**

Python >= 3.7; all analyses and testing were performed using Python 3.7 within PyCharm V.2023.2

Software packages and tools used for all analyses and testing are as follows with links to repositories:

  - sklearn(scikit-learn) V.1.2.2 (https://github.com/scikit-learn/scikit-learn)
     - sklearn.metrics.explained_variance_score
     - sklearn.metrics.normalized_mutual_info_score 
     - sklearn.decomposition.NMF
     - sklearn.decomposition.PCA
     - sklearn.model_selection.train_test_split
     - sklearn.ensemble.GradientBoostingClassifier
  - scipy V.1.10.1 (https://github.com/scipy)
     - scipy.signal 
     - scipy.stats
  - networkx V.2.6.3 (https://github.com/networkx) 
  - scanpy V.1.9.3  https://github.com/scverse/scanpy
     - scanpy.pp.normalize_total
     - scanpy.tl.dpt 
  - pyreadr V.0.4.7 (https://github.com/ofajardo/pyreadr)
  - powerlaw V.1.5 (https://pypi.org/project/powerlaw/)
  - anndata V.0.8.0 (https://github.com/scverse/anndata)
  - matplotlib V.3.5.3 (https://matplotlib.org/)
  - h5py V.3.8.0  (https://github.com/h5py)
  - numpy V.1.21.6 (https://github.com/numpy)
  - pandas V.1.1.5 (https://github.com/pandas-dev)
  - seaborn V.0.11.1 (https://github.com/mwaskom/seaborn)
  - scprep V.1.2.3 (https://github.com/KrishnaswamyLab/scprep)
  - Seurat V.4.0.0 (https://github.com/satijalab/seurat)
  - STutility V.0.1.0 (https://ludvigla.github.io/STUtility_web_site/)
  - PHATE (https://github.com/KrishnaswamyLab/PHATE)
  - CARD (https://github.com/YingMa0107/CARD)

## **5. 📄 Citation & Associated Publication**

We kindly ask you to **📌 cite** our paper if you use our code in your research.

**📘 Publication:**  
Emery BA, Hu X, Klütsch D, Khanzada S, Larsson L, Dumitru I, Frisén J, Lundeberg J, Kempermann G & Amin H. (2025).  
**MEA‑seqX**: High‑Resolution Profiling of Large‑Scale Electrophysiological and Transcriptional Network Dynamics  
*Advanced Science*, 12(20):2412373.  
[🔗 View online](https://doi.org/10.1002/advs.202412373) • [📄 Download PDF](https://github.com/HayderAminLab/MEA-seqX/raw/main/Emery%20et%20al%202025_MEA%E2%80%90seqX.pdf)

<pre>
@article{Emery2025MEAseqX,
  author    = {Brett A. Emery, Xin Hu, Diana Klütsch, Shahrukh Khanzada, Ludvig Larsson, Ionut Dumitru, Jonas Frisén, Joakim Lundeberg, Gerd Kempermann and Hayder Amin},
  title     = {{MEA‑seqX}: High‑Resolution Profiling of Large‑Scale Electrophysiological and Transcriptional Network Dynamics},
  journal   = {Advanced Science},
  volume    = {12},
  number    = {20},
  pages     = {2412373},
  year      = {2025},
  doi       = {10.1002/advs.202412373},
  url       = {https://doi.org/10.1002/advs.202412373}
}
</pre>

## **6. 📬 Contact**

For questions about the 🧠 **`code`**, please [open an issue](https://github.com/HayderAminLab/DENOISING/issues) in this repository.

For questions about the 📄 **`paper`**, feel free to contact  
**✉️ [Dr.-Ing. Hayder Amin](mailto:hayder.amin@dzne.de)** 
