# 3d_gans

## Requirement

### Dependency

The requirement.txt contains all dependencies - main required libraries are:

```
- tensorflow
- numpy
- scipy
- matplotlib
- mpl_toolkits
- papermill
```

### Dataset

- The jupyter notebook gan3d-progressive-example-injupyter-visualization downloads the training data and pretrained model in the first cell automatically from S3
- Pretrained models are available on BitBucket
- Dataset is available on S3 https://s3-eu-west-1.amazonaws.com/bsopenbucket/e4040/data.zip

## Data structure

### Data
Folder with the training data (downloaded automatically)

### Structure
```
- alphaxenas contains all helper function and model
-- controller_coach.py defines coach
-- controller_lstm.py defines LSTM network for coach
-- controller_mcts.py defines MCTS for coach
-- child_model.py defines CNN network as child
-- data_utils.py contains data loading and data augmentation
-- model_utils.py contains helper functions
- data contains data for training
- exp contains output of experiments
- tensorboard_logs contains the logs for tensorboard
- AlphaXENAS.ipynb is main jupyter notebook for training coach and documents all results
- DenseNet.ipynb is a jupyter notebook, which implemented DenseNet (Benchmark) (as this is not the main project this is just a proof-of-concept
```

### Running the model

Training takes upto 4 days for one model, therefore papermill was used for running AlphaXENAS.ipynb 

```
- papermill AlphaXENAS.ipynb AlphaXENAS_output.ipynb 
```
