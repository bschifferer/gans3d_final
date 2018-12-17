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

### Model
```
- Gan3D contains all helper function and model
-- progressive_model.py defines discriminator, generator and up/downblock
-- progressive_training.py defines the training process
-- progressive_visualization.py defines the visualization and testing functions
-- utils.py contains helper function for loading the data and/or visualizing them
```

### Running the model
```
- gan3d-progressive-example-injupyter-visualization.ipynb is an example implementation for a 32x32x32 model
```

Alternative models can bve found in folder: used_jupyters

### Results
Results are stored in the folder /output/<model name>:
```
-- loss contains plots for loss and accuracy
-- model contains snapshots of the model 
-- vis contains visualization 
```

### Previous training
```
- used_jupyters contains previous jupyter notebook, which saves files to disk
-- gan3d-progressive_final_16px_15001.ipynb training a 16x16x16 model
-- gan3d-progressive_final_32px_15001.ipynb training a 32x32x32 model
-- gan3d-progressive_final_64px_15001.ipynb training a 64x64x64 model
-- gan3d-progressive_progressive_16px_64px_2ndrund.ipynb training a progressive grown model from 16px to 64px
-- gan3d_refactored_progressive_final_32px_chair_table_15001_3.ipynb training a model with two classes chair and table for 15000 iterations
-- gan3d_refactored_progressive_final_32px_chair_table_15001_3_continue.ipynb continue training for another 15000 iterations
-- gan3d-progressive-test.ipynb example for plotting resuls
-- gan3d-progressive-test-linear-interpolation.ipynb example for linear interpolation
```

# Training
Training time depends on the resolution:
```
- 16x16x16 requires 1 minute per 200 iterations with batch size 64
- 32x32x32 requires 5 minute per 200 iterations with batch size 64
- 64x64x64 requires 20 minute per 200 iterations with batch size 64
```

A good training method is using papermill e.g.
```
# First, the jupyter notebook needs to be copied in the root directory
papermill gan3d-progressive_final_32px_15001.ipynb gan3d-progressive_final_32px_15001_output.ipynb
```

