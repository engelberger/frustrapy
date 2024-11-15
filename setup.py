from setuptools import setup, find_packages

setup(
    name="frustrapy",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "biopython",
        "plotly",
        "scipy",
        "scikit-learn",
        "python-igraph",
        "leidenalg",
    ],
    extras_require={
        "dev": [
            "psutil",
            "kaleido",
        ],
    },
    author="Felipe Engelberger",
    author_email="felipeengelberger@gmail.com",
    description="A Python package for protein frustration analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/engelberger/frustrapy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
