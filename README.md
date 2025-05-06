# How to run the code

The first thing to focus on is importing the data. We imported the data from Kaggle, using the command:
import kagglehub
path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
 This command stores the data at location: path. Note that the following code might have to be modified to accommodate the user's given path. We then moved the data to the present working directory using the command: mv path/* .

The remainder of the code is run simply by going through the python notebook sequentially. 

Note that you will need to have access to a GPU to run this code. 
Note that some of the graphs may vary as the model's final weights depend on the random initialization of the weights. 

# Required dependencies
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display

import librosa
import IPython.display as ipd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from joblib import Parallel, delayed


# Describe any pretrained models or datasets used
The dataset used in GTZAN, which contains 1000 audio clips (10 genres, 100 clips each), each 30 seconds long, at 22050Hz, and is the standard benchmark for music genre classification.
We used wav2vec, a pretrained audio transformer designed by Facebook to embed the audio files. However, we quickly veered away from this and used mel spectrograms instead.

