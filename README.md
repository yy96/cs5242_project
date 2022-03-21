# Protein-Ligand binding prediction

This is the code base for CS5242 project where we are required to predict ligand and protein binding. The source code can be find in `cs5242_project/code` folder.
- `data.py`: this file defines the data generation process for the project where the test, validation and training datasets are generated
- `feature.py`: this file defines various feature engineering for SMILES string and protein pdb files
- `model_utils.py`: this file provides utility function for the model, mainly the validation scoring function
- different versions of `model.py` files: these files define the model structure tested
- `train.py`: the main training file to trigger the training process
