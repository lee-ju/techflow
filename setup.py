from setuptools import setup

setup(
    name='techflow',
    version='0.0.6',
    description='TechFlow: for Technology Analysis',
    url='https://github.com/lee-ju/techflow.git',
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
