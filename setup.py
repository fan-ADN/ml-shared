from setuptools import setup, find_packages
from pathlib import Path


if __name__ == '__main__':
    setup(
        name='ml_shared',
        version='0.1',
        url='https://github.com/fan-ADN/ml-shared',
        license='Confidential',
        author='s_katagiri',
        author_email='s_katagiri@fancs.com',
        description='',
        packages=find_packages(include='ml_shared'),
        # packages=['ml_shared', 'ml_shared.data', 'ml_shared.model', 'ml_shared.feature', 'ml_shared.evaluation'],
        python_requires='>3.0, <4',
        install_requires=[
            'numpy>=1.17.0',
            'scipy>=1.3.2',
            'pandas>=0.24',
            'matplotlib>=3.1.1',
            'plotnine>=0.6.0',
            'scikit-learn>=0.21.2'
        ],
        classifiers=[
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7'
        ]
    )
