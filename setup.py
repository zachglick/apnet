from setuptools import setup

short_description = "AP-Net is an atomic-pairwise neural network framework for modeling interaction energies"

try:
    with open("README.md", "r") as fp:
        long_description = fp.read()
except FileNotFoundError:
    long_description = short_description

setup(
    name="apnet",
    version="0.0.1",    
    description="Atomic-Pairwise Neural Network for Interaction Energies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Zachary L. Glick",
    author_email="zlg@gatech.edu",
    url="https://github.com/zachglick/apnet",
    packages=["apnet"], # TODO: is it better to use setuputils.find_packages() ?
    classifiers=[ # TODO: is this complete? does that matter?
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",  
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    license="MIT",
    platforms=["any"], # TODO: is this true?
    package_data={"apnet": ["data/*.pkl",
                            "atom_models/*.hdf5",
                            "pair_models/*.h5",
                            ]
                 },
    install_requires=["numpy",
                      "scipy",
                      "qcelemental",
                      "tensorflow>=2.2,<2.4",
                      ],
    python_requires=">=3.7", # TODO: is this too loose?
)
