"""Setup moneyball."""
from setuptools import setup, find_packages
from pathlib import Path
import typing

readme_path = Path(__file__).absolute().parent.joinpath('README.md')
long_description = readme_path.read_text(encoding='utf-8')


def install_requires() -> typing.List[str]:
    """Find the install requires strings from requirements.txt"""
    requires = []
    with open(
        Path(__file__).absolute().parent.joinpath('requirements.txt'), "r"
    ) as requirments_txt_handle:
        requires = [
            x
            for x in requirments_txt_handle
            if not x.startswith(".") and not x.startswith("-e")
        ]
    return requires


setup(
    name='moneyball',
    version='0.0.183',
    description='A library for determining what bets to make.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='sports betting',
    url='https://github.com/8W9aG/moneyball',
    author='Will Sackfield',
    author_email='will.sackfield@gmail.com',
    license='MIT',
    install_requires=install_requires(),
    zip_safe=False,
    packages=find_packages(),
    entry_points = {
        'console_scripts': ['moneyball=moneyball.__main__:main'],
    },
)
