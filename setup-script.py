from setuptools import setup, find_packages

setup(
    name="embolism-prediction",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-Powered Early Detection of Embolism Using Multimodal Clinical Data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/embolism-prediction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.60.0",
    ],
)
