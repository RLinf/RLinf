# read the contents of your README file
from os import path

from setuptools import find_packages, setup

setup(
    name="liberoplus",
    packages=[package for package in find_packages() if package.startswith("liberoplus")],
    install_requires=[],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="liberoplus-plus: In-Depth Robustness Analysis For Vision-Language-Action Models",
    author="Anonymous",
    author_email="Anonymous",
    version="0.1.0",
    long_description="liberoplus-plus",
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "lifelong.main=liberoplus.lifelong.main:main",
            "lifelong.eval=liberoplus.lifelong.evaluate:main",
            "liberoplus.config_copy=scripts.config_copy:main",
            "liberoplus.create_template=scripts.create_template:main",
        ]
    },
)
