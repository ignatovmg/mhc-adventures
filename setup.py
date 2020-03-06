from setuptools import setup, Extension, find_packages
import numpy as np

#cpp_ext = Extension('mhc_adventures.molgrid',
#                    sources=['mhc_adventures/source/molgrid/py_molgrid.cpp'],
#                    include_dirs=[np.get_include()])

setup(name='mhc_adventures',
      version='0.1',
      description='Process MHC structures and train predictors',
      url='https://github.com/ignatovmg/mhc-adventures',
      author='Mikhail Ignatov',
      author_email='ignatovmg@gmail.com',
      license='MIT',
      packages=['mhc_adventures'],
      include_package_data=True,
      zip_safe=False) #,
      #ext_modules=[cpp_ext])
