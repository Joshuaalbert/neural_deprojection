#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

__minimum_numpy_version__ = '1.16.2'

setup_requires = ['numpy>=' + __minimum_numpy_version__]

setup(name='neural_deprojection',
      version='0.0.1',
      description='Neural network based deprojection of projections of star clusters and galaxy clusters',
      author=['Joshua G. Albert', 'Matthijs van Groeningen', 'Julius Hendrix'],
      author_email=['albert@strw.leidenuniv.nl','mvgroeningen@strw.leidenuniv.nl', 'hendrix@strw.leidenuniv.nl'],
      setup_requires=setup_requires,
      tests_require=[
          'pytest>=2.8',
      ],
      # package_data= {'data':['arrays/*', 'data/*']},
      package_dir={'': './'},
      packages=find_packages('./')
      )
