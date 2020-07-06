from setuptools import setup, find_packages


def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()

setup(
    name='models',
    packages=find_packages(exclude=['training', 'movie scraper']),
    package_data={'': ['*', 'data/*', 'SVM/pipelines/*']},
    include_package_data=True,
    install_requires=list_reqs()[1:],
    extras_require={}
)
