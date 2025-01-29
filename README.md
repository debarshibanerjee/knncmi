# Conditional Mutual Information Estimation with Nearest Neighbors

This python code estimates conditional mutual information (CMI) and mutual information (MI) for discrete and/or continuous variables using a nearest neighbors approach.
The background and theory can be found on [arXiv](https://arxiv.org/abs/1912.03387).

## Getting started

Once inside the git repository directory, use the package manager [pip](https://pip.pypa.io/en/stable/) to install `knncmi`.

```bash
pip install .
```

## Usage

`cmi` is the primary function for this package;
it estimates the conditional mutual information or mutual information between random variables or vectors, CMI(X,Y|Z), where X,Y, and Z are lists, `<xlist>, <ylist>, <zlist>`, respectively. 
All other functions are auxiliary functions.
The basic setup is `cmi(<xlist>, <ylist>, <zlist>, k, data, discrete_dist = 1, minzero = 1)`.
- `<xlist>`, `<ylist>`, and  `<zlist>` are lists of variables. They can either be variable names as a list of strings, or a list of the variable indices.
- `k` is a positive integer corresponding to the hyperparameter for the `k`th nearest neighbor.
- `data` is a pandas dataframe.
- `discrete_dist` is the scalar distances specified between distinct, non-numeric, categorical variables. If no value is specified, `1` is used as default.
- `min_zero` indicates if zero should be the smallest value returned by the `cmi` function. When `min_zero=1` or is not specified, the smallest value `cmi` returns will be `0`, otherwise, it may be negative. CMI is non-negative in theory but due estimation, `cmi` can return a negative number.
- If `<zlist>` is empty, `cmi` computes MI, rather than CMI, but the function always expects a list, empty or not.

The code below shows how to use the `knncmi` package and `cmi` function.

```python
import knncmi as k
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Read dataset to pandas dataframe (Iris dataset)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['slength', 'swidth', 'plength', 'pwidth', 'class']
dataset = pd.read_csv(url, names=names)

# estimate CMI between 'slength' and 'swidth' given 'class'
cmi_result = k.cmi(['slength'], ['swidth'], ['class'], 3, dataset)
print(f"CMI(slength, swidth | class): {cmi_result}")
# 0.24623572430840526

# estimate MI between 'class' and 'swidth'
mi_result = k.cmi(['class'], ['swidth'], [], 3, dataset)
print(f"MI(class, swidth): {mi_result}")
# 0.21378704289138606

# Example with a periodic variable
n = 100
theta = np.random.uniform(0, 2*np.pi, n)
x = np.cos(theta) + np.random.normal(0, 0.1, n)
y = np.sin(theta) + np.random.normal(0, 0.1, n)
z = np.random.normal(0, 1, n)
df_periodic = pd.DataFrame({'theta': theta, 'x': x, 'y': y, 'z': z})

periodic_vars = {'theta': 2*np.pi}  # Specify 'theta' as periodic

mi_xy = k.cmi(['x'], ['y'], [], 4, df_periodic, periodic_vars=periodic_vars)
cmi_xy_theta = k.cmi(['x'], ['y'], ['theta'], 4, df_periodic, periodic_vars=periodic_vars)
cmi_xy_z = k.cmi(['x'], ['y'], ['z'], 4, df_periodic, periodic_vars=periodic_vars)

print(f"MI(x, y): {mi_xy}")
# 0.788970661696498

print(f"CMI(x, y | theta): {cmi_xy_theta}")
# 0.03034523809523808

print(f"CMI(x, y | z): {cmi_xy_z}")
# 0.32604131072549014
``` 

## Simulations

The arXiv paper mentioned above includes simulations to empirically show the behavior of this method compared to other, similar methods.
This section shows how to run the python scripts to generate the images in the paper.

In order for this code to run, all packages must already be installed; they can be found at the top of the scripts themselves.
The full simulations for the paper, uses 100 datasets for each scenario.
On an 8-core machine, it takes about 30 mins to run.
To save an unsuspecting person for accidentally tying up their machine, I've altered the original code to run 10 datasets, taking about 3 mins.
To change to the original, code, go to line 19 in the file and change `n = 10` to `n = 100`.

```bash
cd simulations
python3 createSimData.py &
python3 createBoxplots.py 
```
After these scripts have run, a pdf with the images should pop up.

## Tests

The tests included here are the tests used during the development of the code.
They may be helpful for further development as well.

```bash
python3 -m pytest
```
