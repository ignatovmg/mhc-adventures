from setuptools import setup, Extension
import numpy as np

cpp_ext = Extension('ilovemhc.molgrid', sources = ['ilovemhc/src/grid_maker/py_molgrid.cpp'], include_dirs=[np.get_include()])

setup(name='ilovemhc',
      version='0.1',
      description='Process MHC structures and train predictors',
      url='https://github.com/ignatovmg/mhc-adventures',
      author='Mikhail Ignatov',
      author_email='mikhail.ignatov@stonybrook.edu',
      license='MIT',
      packages=['ilovemhc'],
      include_package_data=True,
      zip_safe=False, 
      ext_modules=[cpp_ext])
