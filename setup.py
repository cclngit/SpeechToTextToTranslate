from setuptools import setup, find_packages

setup(
    name='speech_processing_pipeline',
    version='1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.20.3',
        'torch>=1.10.0',
        'transformers>=4.11.3'
    ],
    entry_points={
        'console_scripts': [
            'run_pipeline=run_pipeline:main',
        ],
    },
)
