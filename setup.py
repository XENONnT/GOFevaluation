import setuptools

# Get requirements from requirements.txt, stripping the version tags
with open("requirements.txt") as f:
    requires = [r.split("/")[-1] if r.startswith("git+") else r for r in f.read().splitlines()]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("HISTORY.md") as file:
    history = file.read()

setuptools.setup(
    name="GOFevaluation",
    version="0.1.5",
    author="GOFevaluation contributors, the XENON collaboration",
    description="Evaluate the Goodness-of-Fit(GOF) for binned or \
        unbinned data.",
    long_description=long_description + "\n\n" + history,
    long_description_content_type="text/markdown",
    setup_requires=["pytest-runner"],
    install_requires=requires,
    tests_require=requires
    + [
        "pytest",
        "flake8",
    ],
    python_requires=">=3.8",
    url="https://github.com/XENONnT/GOFevaluation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
