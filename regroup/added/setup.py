from setuptools import setup, find_packages

setup(
    name="regroup",
    version="0.1.0",
    packages=find_packages(where="regroup"),
    package_dir={"": "regroup"},
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "opencv-python",
        "Pillow",
    ],
)
