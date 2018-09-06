"""setup.py file for packaging ``swagexample``."""

from setuptools import setup, find_packages


with open('README.md', 'r') as readme_file:
    readme = readme_file.read()


setup(
    name='swagaf',
    version='0.0.0',
    description="A baseline submission for the SWAG leaderboard.",
    long_description=readme,
    keywords='deep learning swag allennlp',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    license='Apache',
    packages=find_packages(),
    install_requires=[
        'allennlp',
        'ipython',
        'torch',
        'torchvision'
    ],
    python_requires='>=3.6',
    zip_safe=False
)