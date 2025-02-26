# Reality Defender Audio Model Interpretation Assessment

This repository contains the code for decision boundary analysis. 

## Installation
First, clone the repository locally:

```
git clone https://github.com/hashim19/Decision_Boundary_Analysis.git
```

Next, create and activate the conda environment:
```
conda create -n "db_analysis" python=3.9.19
conda activate db_analysis
```
Then, install the requirements:
```
pip3 install -r requirements.txt
```

### Descision Boundary analysis

To run decision boundary analyzer, run:

```
python3 descision_boundary_analysis.py --config ./rd_assesment.yaml
```

This will load the trained model, computes the umap embeddings on the ASVspood 2019 LA eval subset, and generates the umap visualization in viusalization directory. This will also generates a csv file which contains all the results. 

### SAMPLE ANALYSIS AND INTERPRETATION

Use model_interpretation.ipynb notebook for model interpretation and generating plots.
