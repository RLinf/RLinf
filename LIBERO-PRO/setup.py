# read the contents of your README file
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "./README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="liberopro",
    packages=[package for package in find_packages() if package.startswith("liberopro")],
    install_requires=[],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="liberopro: Benchmarking Knowledge Transfer for Lifelong Robot Learning",
    author="Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone",
    # url="https://github.com/ARISE-Initiative/robosuite",
    author_email="bliu@cs.utexas.edu, yifengz@cs.utexas.edu",
    version="0.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "lifelong.main=liberopro.lifelong.main:main",
            "lifelong.eval=liberopro.lifelong.evaluate:main",
            "liberopro.config_copy=scripts.config_copy:main",
            "liberopro.create_template=scripts.create_template:main",
        ]
    },
)
