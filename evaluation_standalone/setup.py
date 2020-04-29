from distutils.core import setup

setup(
    name='codesearchnet',
    version='0.1a',
    python_requires='>3.6',
    description='Evaluate the codesearchnet challenge with this package',
    author='',
    author_email='masterprojekt@hs-rm.de',
    license='gnu',
    packages=['codesearchnet'],
    install_requires=[
      'numpy', 'torch', 'tqdm', 'annoy'
    ]
)
