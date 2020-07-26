from setuptools import setup

setup(name='rubin_rotation',
      version='0.1rc0',
      description='Tools for rotation with Rubin',
      url='http://github.com/RuthAngus/rubin_rotation',
      author='Ruth Angus',
      author_email='ruthangus@gmail.com',
      license='MIT',
      packages=['rubin_rotation'],
      include_package_data=True,
      install_requires=['numpy', 'exoplanet', 'pymc3', 'theano'],
      zip_safe=False)
