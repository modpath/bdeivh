import os

from setuptools import setup, find_packages

setup(
    name='bdeivh',
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    package_data={'bdeivh': [os.path.join('', 'README.md'),
                                  os.path.join('models', '*.json'),
                                  os.path.join('models', '*.weights.h5'),
                                  os.path.join('', 'LICENCE')]},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    version='0.1.1',
    description='Estimation of BDEIVH parameters from phylogenetic trees.',
    maintainer='Anna Zhukova',
    maintainer_email='anna.zhukova@pasteur.fr',
    author='Ighor Arantes',
    author_email='ighor.arantes@pasteur.fr',
    url='https://github.com/modpath/bdeivh',
    keywords=['phylogenetics', 'birth-death model', 'incubation', 'arbovirus', 'human-vector-human transmission'],
    install_requires=['tensorflow==2.13.1', 'pandas==2.2.3', 'ete3>=3.1.3', 'numpy>=1.23.5,<2.0.0',
                      'six>=1.16.0', 'matplotlib', 'treesumstats>=1.1', 'scikit-learn',
                      ],
    entry_points={
            'console_scripts': [
                'bdeivh_infer = bdeivh.Genomic_tree_predictions_temporal.py:main',
            ]
    },
)
