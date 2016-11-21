import numpy as np


def partmask(zm, frac):
    z = zm
    frac = frac*100
    for i in range(0, z.shape[0]):
        if z[i] == True:
            if np.random.randint(100) < frac:
                z[i] = False
    return z
