from setuptools import setup, find_packages

import lane_lines

setup(
    name='lane_lines',
    version=lane_lines.__version__,
    packages=find_packages(),
    long_description=open('README.md').read(),
    license=open('LICENSE').read(),
    entry_points={'console_scripts' : ['run_lane_lines = lane_lines.core:run']}
    )