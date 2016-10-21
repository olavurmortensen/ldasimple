#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(name='LdaSimple',
      version='1.0',
      description='Simple implementation of batch and online variational Bayes for Latent Dirichlet Allocation.',
      author='Ólavur Mortensen',
      author_email='olavurmortensen@gmail.com',
      packages=find_packages(),
     )
