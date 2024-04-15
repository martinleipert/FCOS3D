from setuptools import setup, find_packages


setup(
    name="fcos3d",
    version="1.0",
    packages=find_packages(),
    requires=["torch", "torchvision", "pytorch3d"]
)

