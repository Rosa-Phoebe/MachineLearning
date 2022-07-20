# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 19:05:22 2022

Reference:
    https://www.sfu.ca/~ssurjano/rosen.html
    https://www.indusmic.com/post/rosenbrock-function
    https://people.bath.ac.uk/ps2106/files/courses/MA40050/2020/jupyter/Rosenbrock.html
    https://scientific-python.readthedocs.io/en/latest/notebooks_rst/5_Optimization/04_Exercices/00_Tutorials/Optimization_Tutorial.html
    https://jamesmccaffrey.wordpress.com/2021/07/27/poking-around-the-rosenbrock-function/
    https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    https://stackoverflow.com/questions/35445424/surface-and-3d-contour-in-matplotlib

@author: 
"""

"""
    Rosenbrock function:
        The Rosenbrock function, also referred to as Valley or Banana function,
        is a popular test problem for gradient-based optimization algorithms.
        The function is unimodal and the global minimum lies in a narrow,
        parabolic valley. However, even though this valley is easy to find,
        convergence to the minimum is difficult.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

b = 10
# Rosenbrock two-dimensional form: dim = 2
# Rosenbrock's banana function: f(x,y)=(1-x)^2+100(y-x^2)^2
rosenbrockfunction = lambda x, y: (x - 1) ** 2 + b * (y - x**2)**2

# Initialize figure: Create an empty figure
figRos = plt.figure(figsize=(12, 7))
# Define axes as 3D axes so plot 3D data into it
axRos = plt.axes(projection="3d")
# axRos = figRos.add_subplot(projection="3d")

# Evaluate function: Input Domain restricted
X = np.linspace(-0.5, 1.5, 100)   # X = np.arange(-2, 2, 0.15)
Y = np.linspace(-1.6, 1.6, 100)   # Y = np.arange(-1, 3, 0.15)
X, Y = np.meshgrid(X, Y)
Z = rosenbrockfunction(X, Y)
Z = np.log10(Z)


# Plot the surface
# Or use cm.gist_heat_r
surf = axRos.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0.5, antialiased=False)
figRos.colorbar(surf, shrink=0.5, aspect=10)

# This contour not show???
axRos.contour(X, Y, Z, 4, cmap=cm.coolwarm)#cmap='gray_r')

# Global Minimum
# axRos.plot([1], [1], 'x', mew=3, markersize=10, color='#111111')
# 

#axRos.set_xlim(-1.3, 1.3)
#axRos.set_ylim(-0.9, 1.7)
#axRos.set_zlim(-60, 60)   # axRos.set_zlim(0, 300)
# axRos.zaxis.set_scale('log')
# Show plot
plt.show()



































































































