import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gLEC",
    version="0.0.1",
    author="Luke Mondy",
    author_email="luke.mondy@sydney.edu.au",
    description="Calculate LEC on triangular irregular meshes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LukeMondy/gLEC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
