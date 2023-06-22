"""
Generate some random samples from a smoothly varying function and then implement and evaluate
one or more data interpolation techniques.
    1. Generate a "random" 1-D or 2-D function by adding together a small number of sinusoids
        or Gaussians of random amplitudes and frequencies or scales.
    2. Sample this function at a few dozen random locations.
    3. Fit a function to these data points using one or more of the scattered data interpolation
        techniques described in Section 4.1.
    4. Measure the fitting error between the estimated and original functions at some set of
        location, e.g., on a regular grid or at different random points.
    5. Manually adjust any parameters your fitting algorithm may have to minimize the output
        sample fitting error, or use an automated technique such as cross-validation.
    6. Repeat this exercise with a new set of random input sample and output sample locations.
        Does the optimal parameter change, and if so, by how much?
    7. (Optional) Generate a piecewise-smooth test function by using different random parameters
        in different parts of your image.
            a. How much more difficult does the data fitting problem become?
            b. Can you think of ways you might mitigate this?
        Try to implement your algorithm in NumPy (or Matlab) using only array operations, in order
    to become more familiar with data-parallel programming and the linear algebra operators
    built into these systems. Use data visualization techniques such as those in Figures 4.3-4.6
    to debug your algorithms and illustrate your results.
"""

import numpy as np
from math import sin
from scipy import stats
import matplotlib.pyplot as plt
import numpy.linalg as la

def base_func(x):
    return stats.norm(2.3, 5.3).pdf(x) * 50 + sin(x)

# f(x) = \sum_k w_k \phi (||x-x_k||)
# every radial function is centered at one datapoint

def sample_error(func, x):
    """
        :param func: callable that's my interpolated func
        :param x: where to compute
    """
    return sum(np.abs(np.array(list(map(base_func, x))) - np.array(list(map(func, x)))))


def scattered_data_interpolation_gaussians(k: int):
    """
       :param k: the number of datapoints or kernels
    """
    x = np.arange(-10, 10, 0.1)
    y = np.array(list(map(base_func, x)))
    sample_x = np.random.rand(k) * 20 - 10
    sample_y = np.array(list(map(base_func, sample_x)))
    scales = np.random.rand(k) * 10

    D = np.tile(sample_x, (len(sample_x), 1)) - sample_x[:, None]
    A = np.exp(-D**2/scales[None, :]**2)
    # A = np.array([[exp(-D[k, i]**2/scales[i]**2) for i in range(len(scales))] for k in range(len(sample_x))])
    w = la.solve(A, sample_y)

    func = lambda x: w @ np.exp(-(x - sample_x)**2/scales**2)

    print(f"sample_error = {sample_error(func, np.random.rand(5)*20 - 10)}")

    plt.plot(x, y)
    plt.plot(x, list(map(func, x)), 'r')
    plt.scatter(sample_x, sample_y, s=80, facecolors='none', edgecolors='g')
    plt.show()


if __name__ == "__main__":
    scattered_data_interpolation_gaussians(10)
