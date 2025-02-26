# Reality Defender Audio Model Interpretation Assessment

This repository contains the code implementaton of Reality Defender Audio Model Interpretation Assessment task. 

## Installation
First, clone the repository locally:

```
git clone https://github.com/hashim19/RD-Audio-Takehome-Hashim-Ali.git
```

Next, create and activate the conda environment:
```
conda create -n "rd_assessment" python=3.9.19
conda activate rd_assessment
```
Then, install the requirements:
```
pip3 install -r requirements.txt
```

## Experiments

### Dataset
This repository uses a sampled version of the ASVspoof 2019 LA databases, which can be downloaded from [here](https://drive.google.com/drive/folders/1G1hVvyU9t4XyDsg3hUXxPG8ed7TmfOu9)

### Training

To train the model, first set the database path of the protocol file and flac files in yaml file, and then run:
```
python3 main.py --config ./rd_assesment.yaml --comment rd_assessment
```

### Evaluation

To evaluate the model, first set the path of the model trained with augmented dataset in yaml file, and then run:
```
python3 main.py --config ./rd_assesment.yaml --eval --comment laundered
```

This will generate a score file in the Score_Files directory and an eer text file which will contain the overall EER and a detailed breakdown of results by attack type.

### Results Using the Laundered Augmented Model

CM SYSTEM

        EER		= 2.400000000 % (Equal error rate for countermeasure)

BREAKDOWN CM SYSTEM

        EER A07		= 0.400000000 % (Equal error rate for A07)
        
        EER A08		= 0.400000000 % (Equal error rate for A08)
	
        EER A09		= 0.000000000 % (Equal error rate for A09)
	
        EER A10		= 0.600000000 % (Equal error rate for A10)
	
        EER A11		= 0.000000000 % (Equal error rate for A11)
	
        EER A12		= 2.715789474 % (Equal error rate for A12)
	
        EER A13		= 0.000000000 % (Equal error rate for A13)
	
        EER A14		= 0.000000000 % (Equal error rate for A14)
	
        EER A15		= 0.400000000 % (Equal error rate for A15)
	
        EER A16		= 2.020408163 % (Equal error rate for A16)
	
        EER A17		= 4.700000000 % (Equal error rate for A17)
	
        EER A18		= 2.463829787 % (Equal error rate for A18)
	
        EER A19		= 3.483333333 % (Equal error rate for A19)

### Descision Boundary analysis

To run decision boundary analyzer, run:

```
python3 descision_boundary_analysis.py --config ./rd_assesment.yaml
```

This will load the trained model, computes the umap embeddings on the ASVspood 2019 LA eval subset, and generates the umap visualization in viusalization directory. This will also generates a csv file which contains all the results. 

### SAMPLE ANALYSIS AND INTERPRETATION

Use model_interpretation.ipynb notebook for model interpretation and generating plots.
