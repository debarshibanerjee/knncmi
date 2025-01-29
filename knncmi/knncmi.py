###  Author: Octavio Mesner
###  This code should be able to compute
###  Conditional Mutual Information using kNN

import numpy as np
import pandas as pd
from scipy.special import digamma

# If a variable is categorical, M/F eg, we must define a distance
# if not equal
# might need to be changed to something else later

def getPairwiseDistArray(data, coords = [], discrete_dist = 1, periodic_vars = {}):
    '''
    Input:
    data: pandas data frame
    coords: list of indices for variables to be used
    discrete_dist: distance to be used for non-numeric differences
    periodic_vars: dictionary with keys as column indices and values as periods

    Output:
    p x n x n array with pairwise distances for each variable
    '''
    n, p = data.shape
    if coords == []:
        coords = range(p)
    col_names = list(data)
    distArray = np.empty([p,n,n])
    distArray[:] = np.nan
    for coord in coords:
        thisdtype=data[col_names[coord]].dtype
        if pd.api.types.is_numeric_dtype(thisdtype):
            diff = abs(data[col_names[coord]].to_numpy() -
                                       data[col_names[coord]].to_numpy()[:,None])
            if coord in periodic_vars:
                period = periodic_vars[coord]
                distArray[coord,:,:] = np.minimum(diff, period - diff)
            else:
                distArray[coord,:,:] = diff
        else:
            distArray[coord,:,:] = (1 - (data[col_names[coord]].to_numpy() ==
                                    data[col_names[coord]].to_numpy()[:,None])) * discrete_dist
    return distArray

def getPointCoordDists(distArray, ind_i, coords = list()):
    '''
    Input:
    ind_i: current observation row index
    distArray: output from getPariwiseDistArray
    coords: list of variable (column) indices

    output: n x p matrix of all distancs for row ind_i
    '''
    if not coords:
        coords = range(distArray.shape[0])
    obsDists = np.transpose(distArray[coords, :, ind_i])
    return obsDists

def countNeighbors(coord_dists, rho, coords = list()):
    '''
    input: list of coordinate distances (output of coordDistList),
    coordinates we want (coords), distance (rho)

    output: scalar integer of number of points within ell infinity radius
    '''

    if not coords:
        coords = range(coord_dists.shape[1])
    dists = np.max(coord_dists[:,coords], axis = 1)
    count = np.count_nonzero(dists <= rho) - 1
    return count

def getKnnDist(distArray, k):
    '''
    input:
    distArray: numpy 2D array of pairwise, coordinate wise distances,
    output from getPairwiseDistArray
    k: nearest neighbor value

    output: (k, distance to knn)
    '''
    dists = np.max(distArray, axis = 1)
    ordered_dists = np.sort(dists)
    # using k, not k-1, here because this includes dist to self
    k_tilde = np.count_nonzero(dists <= ordered_dists[k]) - 1
    return k_tilde, ordered_dists[k]

def cmiPoint(point_i, x, y, z, k, distArray):
    '''
    input:
    point_i: current observation row index
    x, y, z: list of indices
    k: positive integer scalar for k in knn
    distArray: output of getPairwiseDistArray

    output:
    cmi point estimate
    '''
    n = distArray.shape[1]
    coord_dists = getPointCoordDists(distArray, point_i, x + y + z)
    k_tilde, rho = getKnnDist(coord_dists, k)
    x_coords = list(range(len(x)))
    y_coords = list(range(len(x), len(x+y)))
    z_coords = list(range(len(x+y), len(x+y+z)))
    nxz = countNeighbors(coord_dists, rho, x_coords + z_coords)
    nyz = countNeighbors(coord_dists, rho, y_coords + z_coords)
    nz = countNeighbors(coord_dists, rho, z_coords)
    xi = digamma(k_tilde) - digamma(nxz) - digamma(nyz) + digamma(nz)
    return xi

def miPoint(point_i, x, y, k, distArray):
    '''
    input:
    point_i: current observation row index
    x, y: list of indices
    k: positive integer scalar for k in knn
    distArray: output of getPairwiseDistArray

    output:
    mi point estimate
    '''
    n = distArray.shape[1]
    coord_dists = getPointCoordDists(distArray, point_i, x + y)
    k_tilde, rho = getKnnDist(coord_dists, k)
    x_coords = list(range(len(x)))
    y_coords = list(range(len(x), len(x+y)))
    nx = countNeighbors(coord_dists, rho, x_coords)
    ny = countNeighbors(coord_dists, rho, y_coords)
    xi = digamma(k_tilde) + digamma(n) - digamma(nx) - digamma(ny)
    return xi

def cmi(x, y, z, k, data, discrete_dist = 1, minzero = 1, periodic_vars = {}):
    '''
    computes conditional mutual information, I(x,y|z)
    input:
    x: list of indices for x
    y: list of indices for y
    z: list of indices for z
    k: hyper parameter for kNN
    data: pandas dataframe
    periodic_vars: dictionary with keys as column indices and values as periods

    output:
    scalar value of I(x,y|z)
    '''
    # compute CMI for I(x,y|z) using k-NN
    n, p = data.shape

    # convert variable to index if not already
    vrbls = [x,y,z]
    for i, lst in enumerate(vrbls):
        if all(type(elem) == str for elem in lst) and len(lst) > 0:
            vrbls[i] = list(data.columns.get_indexer(lst))
    x,y,z = vrbls

    distArray = getPairwiseDistArray(data, x + y + z, discrete_dist, periodic_vars)
    if len(z) > 0:
        ptEsts = map(lambda obs: cmiPoint(obs, x, y, z, k, distArray), range(n))
    else:
        ptEsts = map(lambda obs: miPoint(obs, x, y, k, distArray), range(n))
    if minzero == 1:
        return(max(sum(ptEsts)/n,0))
    elif minzero == 0:
        return(sum(ptEsts)/n)

def main():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['slength', 'swidth', 'plength', 'pwidth', 'class']
    df = pd.read_csv(url, names=names)
    print(cmi(['class'],['swidth'],['plength'], 4, df))

    # estimate CMI between 'slength' and 'swidth' given 'class'
    print(cmi(['slength'], ['swidth'], ['class'], 3, df))
    # 0.2653312593213504

    # estimate MI between 'class' and 'swidth'
    print(cmi(['class'], ['swidth'], [], 3, df))
    # 0.24637878408866076

    # Example with periodic variable
    # Set seed for reproducibility
    np.random.seed(42)
    n = 100
    theta = np.random.uniform(0, 2*np.pi, n)
    x = np.cos(theta) + np.random.normal(0, 0.1, n)
    y = np.sin(theta) + np.random.normal(0, 0.1, n)
    z = np.random.normal(0, 1, n)
    df_periodic = pd.DataFrame({'theta': theta, 'x': x, 'y': y, 'z': z})

    periodic_vars = {'theta': 2*np.pi}  # Specify 'theta' as periodic with period 2*pi

    print("PERIODIC:")
    print(cmi(['x'],['y'],[], 4, df_periodic, periodic_vars = periodic_vars)) # MI(x,y)
    print(cmi(['x'],['y'],['theta'], 4, df_periodic, periodic_vars = periodic_vars)) # CMI(x,y|theta)
    print(cmi(['x'],['y'],['z'], 4, df_periodic, periodic_vars = periodic_vars)) # CMI(x,y|z)


if __name__ == '__main__':
    main()
