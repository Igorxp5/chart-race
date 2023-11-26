__author__ = 'Igorxp5'

__license__ = 'MIT'
__version__ = '1.0.0'

import setuptools

with open('README.md', 'r', encoding='utf-8') as file:
    long_description = file.read()

setuptools.setup(
    name='chart-race',
    version=__version__,
    author=__author__,
    description='Generate a chart race video',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Igorxp5/chart-race',
    packages=['chart_race', 'chart_race.console_scripts'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': ['chart-race=chart_race.console_scripts.__main__:main'],
    },
    install_requires=['Pillow~=9.3.0', 'tqdm~=4.48.0', 'opencv-python~=4.6.0', 'numpy~=1.24.0'],
    python_requires='>=3.8',
)
