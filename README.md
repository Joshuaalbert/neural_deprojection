# neural_deprojection
Using neural networks to deprojection astronomical observables

# Install

Make a conda environment for this project. I'll call it `tf_py` because it will contain tensorflow.
``` bach
conda create -n tf_py python=3.8
```
Activate `tf_py`
```bash
conda activate tf_py
```
Install all required packages
```bash
pip install numpy tensorflow tensorflow_probability matplotlib scipy pytest
```

Note that AMUSE may require it's own special environment, with differences from this setup. 
You should still be able to follow the same method of modularising that environment.

# Set up pycharm professional

Clone the package
```bash
git clone https://github.com/Joshuaalbert/neural_deprojection.git
```
Make a new project using `.../neural_deprojection` as the project path.
Choose `tf_py` as your interpreter (verify it's in use after making the project).
Go to Settings (Ctrl-Alt-s) then Tools>Python Integrate Tools and select `pytest` as default test runner and `Google` as docstring format.
Go to Settings>Tools>Python Scientific and make sure show plots in tool window is checked.
When you commit for first time you'll need to enter github login.
