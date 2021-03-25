# Logistic Regression
![enter image description here](https://i0.wp.com/post.healthline.com/wp-content/uploads/2020/06/485800-Heart-Disease-Facts-Statistics-and-You-1296x728-Header.png?h=1528)

In Machine Learning, Logistic Regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable. Unlike Linear Regression models which use continuous data, Logistic Regression models are able to use categorical datasets and explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.

# Project Description
This project employs a Logistic Regression model with the objective to predict the risk of Cardiovascular Disease based on 16 pateitn variables such as age, waist citcumference, and preexisting health conditions. First, a binary classification model is created and optimized to predict whether risks are present. Next, the coefficients of all variables are extracted and ordered by importance to understand which factors most influence the development of heart disease. Lastly, the model's performance is evaluated using measures including the Accuracy score, Precision, Recall, and AUC score.


## Steps

 1. Building and optimizing Logistic Regression model
 2. Extracting Features and their Influence
 3. Performance Evaluation

## Requirements

**Python.** Python is an interpreted, high-level and general-purpose programming language. 

**Integrated Development Environment (IDE).** Any IDE that can be used to view, edit, and run Python code, such as:
- [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)
- [Jupyter Notebook](https://jupyter.org/).

### Packages 

Install the following packages in Python prior to running the code.
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
```

## Launch
Download the Python File *CA05-A_Logistic_Regression* and open it in the IDE. Download and import the dataset *cvd_data.csv*. 

## Authors

[Silvia Ji](https://www.linkedin.com/in/silviaji/) - [GitHub](github.com/jisilvia)

## License
This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License.

## Acknowledgements

The project template and dataset provided by [Arin Brahma](https://github.com/ArinB) at Loyola Marymount University.
