# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from setuptools import setup

setup(
    name='techflow',
    version='0.0.3',
    description='TechFlow: for Technology Analysis',
    url='https://github.com/lee-ju/dev_techflow.git',
    author='Juhyun Lee',
    author_email='leeju@korea.ac.kr',
    license='Juhyun Lee',
    packages=['techflow'],
    zip_safe=False,
    install_requires=[
        'networkx',
        'gensim',
        'nltk',
    ]
)
