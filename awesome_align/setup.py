from setuptools import setup, find_packages
import codecs

setup(
    name='awesome_align',
    install_requires=[
        'tokenizers>=0.5.2',
        'torch>=1.2.0',
        'tqdm',
        'numpy',
        'boto3',
        'filelock',
        'requests',
        'urllib3<1.27,>=1.25.4'
    ],
    version='0.1.7',
    author='NeuLab',
    author_email='zdou0830@gmail.com',
    license='BSD 3-Clause',
    description='An awesome word alignment tool',


    long_description=codecs.open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "awesome-align-codet5=awesome_align.run_align_codet5:main",
            "awesome-align-coderosetta-encoder-decoder=awesome_align.run_align_coderosetta_encoder_decoder:main",
        ],

    },
    url='https://github.com/neulab/awesome-align',
)
