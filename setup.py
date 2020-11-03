import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="HL7reporting",
    version="0.0.8",
    author="PJ Gibson",
    author_email="pjgibson25@gmail.com",
    description="A package designed to improve HL7 Data Quality reporting in the field of public health informatics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pjgibson25/locate-nssp-elements",
    packages=setuptools.find_packages(include=["HL7reporting"]),
    package_data={'HL7reporting': ['supporting/*.xlsx']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
