from setuptools import setup, find_packages

setup(
    name="WhiSQA",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
    ],
    author="Your Name",
    description="My awesome package",
    python_requires=">=3.8",
)