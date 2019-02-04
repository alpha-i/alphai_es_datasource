from setuptools import setup
from setuptools import find_packages


setup(
    name='alphai_es_datasource',
    version='0.0.7',
    description='datasource using elasticsearch',
    author='Daniele Murroni, Gabriele Alese',
    author_email='daniele.murroni@alpha-i.co, gabriele.alese@alpha-i.co',
    packages=find_packages(exclude=['doc', 'tests*']),
    install_requires=[
        'elasticsearch',
        'elasticsearch-dsl',
        'pandas==0.22',
        'requests',
        'keras'
    ],
    dependency_links=[]
)
