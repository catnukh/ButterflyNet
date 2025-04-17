# 🦋 ButterflyNet 🦋
## About project
This project is focused on classifying different spesies of butterflies using deep learning technics. The classification will be performed using a convolutional neural network (CNN) implemented with the PyTorch library. And final pretreined model ResNet18.
## How to use
### Intall libraries

Here is the list of libraries needed:
* PyTorch
* scikit-learn
* Pandas
* PIL
* Optuna
* MLflow
* tqdm
* matplotlib
* seaborn
* zipfile

## Download dataset
Since the dataset is too large to be stored on GitHub, you can download it manually from Kaggle:
[Page with dataset](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification/data)

You need to click here:
<img width="880" alt="Знімок екрана 2025-03-27 132757" src="https://github.com/user-attachments/assets/c93fcba3-fb2d-48ef-8210-ab96d2bc63f2" />
And then here:
<img width="428" alt="image" src="https://github.com/user-attachments/assets/e21e323c-6da1-4245-a9af-92ba586fbb5d" />
Then choose folder where you will download project.

## Navigation
```preprocessing.ipynb``` --- downloaded dataset, provided augmentation, tensored, explored features.

```collect_data.py``` --- saved dataloaders.

```basic_model``` --- built different versions of CNNs, tried to find optimal architecture to prevent overfting and underfiting.

```hyperparams.ipynb``` --- tried to find optimal optimizer, learning rate and dropout rate to encrease acuraccy and F1-score, minimize loss.

```pretreined.ipynb``` --- applied transfer learning with ResNet18, progressively unfroze layers, tuned learning rate and augmentations to improve generalization.





