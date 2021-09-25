from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    # Self-descriptive entries which should always be present
    name='BBMM',
    author='Jiace Sun',
    author_email='jsun3@caltech.edu',
    description="BBMM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SUSYUSTC/BBMM",
    license='Open Source',

    # What packages are required for install
    install_requires=[],
    extras_require={
        'tests': [
            'unittest',
        ],
    },
    packages=["BBMM",
              "BBMM.kern",
              "BBMM.regression"],
)
