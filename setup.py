import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ADTdq",
    version="1.1.2",
    author="PJ Gibson",
    author_email="Peter.Gibson@doh.wa.gov",
    description="A package designed to improve HL7 ADT Data Quality reporting in the field of public health informatics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pjgibson25/ADTdq",
    packages=setuptools.find_packages(include=["ADTdq"]),
    package_data={'ADTdq': ['supporting/*.xlsx']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
