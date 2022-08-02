# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 19:56:31 2022

Reference:
    Colab Widgets
    
    
    Colaboratory's widgets enable redirecting future output to a particular
    place in the layout rather than requiring users to provide preconstructed
    html.
    
    Widgets also support a simple protocol for updating and clearing outputs
    dynamically.

@author: 
"""

from google.colab import widgets
from goolge.colab import output
from six.moves import zip


import numpy as np
import random
import time
from matplotlib import pylab





# Now we can create a grid, optional `header_row` and `header_column` control
# whether we want header elements in the grid
# Any output can be printed to individual cells, not just print statements.
grid = widgets.Grid(2, 2, header_row=True, header_column=True)
with grid.output_to(1, 1):
    print("Bye grid")

# Note we can output to arbitrary cell, not necessarily in order
with grid.output_to(0, 0):
    print("Hello grid")
    
print("Now we are outside")

with grid.output_to(1, 0):
    print("Back inside!")
    


# Dynamic data population
# Individual cells can be cleared as new data arrives.
grid = widgets.Grid(2, 2)
for i in range(20):
    with grid.output_to(random.randint(0, 1), random.randint(0, 1)):
        grid.clear_cell()
        pylab.figure(figsize=(2, 2))
        pylab.plot(np.random.random((10, 1)))
    time.sleep(0.5)



# `TabBar` provide a tabbed UI that shows one group of outputs among several
# while hiding the rest. All outputs can be sent to tabs: simple print(),
# matplotlib plots, or any of the rich interactive libraries' outputs.
def create_tab(location):
    tb = widgets.TabBar(['a', 'b'], location=location)
    with tb.output_to('a'):
        pylab.figure(figsize=(3, 3))
        pylab.plot([1, 2, 3])
    
    # Note you can access tab by its name (if they are unique) or by its index
    with tb.output_to(1):
        pylab.figure(figsize=(3, 3))
        pylab.plot([3, 2, 3])
        pylab.show()

print('Different orientations for tabs')
positions = ['start', 'bottom', 'end', 'top']

for p, _ in zip(positions, widgets.Grid(1, 4)):
    print('---- %s ----' % p)
    create_tab(p)

# `TabBar.clear_tab`, like Grid, Tabbar supports clearing individual tabs.
t = widgets.TabBar(["hi", "bye"])
with t.output_to(0):
    print("I am temporary")
    t.clear_tab() # clear current
    print("I am permanent")
    
with t.output_to(1):
    print("Me too temporary")
    
# Clear works outside of with statement
t.clear_tab(1)

with t.output_to(1):
    print("1/2 Me is permanent")
with t.output_to(1):
    print("2/2 Me too is permanent")



# Iterating over a widget
grid = widgets.Grid(3, 3)
for i, (row, col) in enumerate(grid):
    print('Plot!')
    pylab.figure(figsize=(2, 2))
    pylab.plot([i, row, col])

# Note: We can re-enter individual cells!
for i, (row, col) in enumerate(grid):
    print('data at cell %d (%d, %d)\n' % (i, row, col))



# Widget Removal
t = widgets.TabBar(["hi", "bye"])
g = widgets.Grid(3, 3)
print("1/4 You also will see me")
print("2/4 see me")

for _ in t:
    print("***** You only see me temporarily!")

print("3/4 I am safe")

for _ in g:
  print("**** Or me for that matter")

print("4/4 You also will see me")

time.sleep(1)
t.remove()
g.remove()