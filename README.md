# AP-Net

AP-Net is a python package for modeling intermolecular interactions with atomic-pairwise neural networks.
The AP-Net architecture provides smooth and asymptotically correct interaction potentials.

## Getting Started

It is strongly recommended that you install this package in some kind of virtual environment (like a conda environment).
Create and activate a new conda environment with the following commands:
```
>>> conda create --name apnet python=3.8
>>> conda activate apnet
```

Next, clone this repository and `cd` into the top level (the level with this README file).
Run the following command to install the `apnet` package (and dependencies) into your current environment:
```
>>> pip install -e .
```

You should now be able to call `import apnet` from any python script
(no further need to work out of this repository).
Verify that everything is working by running the following python code snippet:
```
import apnet
import qcelemental as qcel

dimer = qcel.models.Molecule.from_data("""
0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
--
0 1
O 3.000000 0.500000  0.000000
H 3.758602 0.500000  0.504284
H 3.260455 0.500000 -0.872893
""")

prediction, uncertainty = apnet.predict_sapt(dimer)
print(prediction)
```

The output should look as follows:
```
[array([-2.87016274, -3.77428398,  2.41356459, -0.50601828, -1.00342507])]
```
AP-Net predicts an interaction energy of -2.87 kcal / mol.
The predicted SAPT depcomposition is -3.77 (electrostatics), +2.41 (exchange), -0.51 (induction) and -1.00 (dispersion).
(For reference, the actual SAPT interaction energy of this dimer is pretty close: -2.66 kcal / mol.
The actual SAPT decomposition is also in good agreement: -3.46 (electrostatics), +2.31 (exchange), -0.53 (induction), and -0.97 (dispersion).)

## Using the Code

The `docs/` directory of this repository contains detailed documentation for using this package.
In the near future, this documentation will be hosted online for easy reference.
For now, you can navigate this local copy of the documentation by opening `docs/build/html/index.html` in a web browser.
Note that the installation instructions in docs are different from the instructions here.
The docs contain install instructions that will take effect when the `apnet` package is made public.




<!--The features of this python package include:-->
<!--* Predict SAPT interaction energies with a pre-trained model-->
<!--* Predict atomic charge distributions with a pre-trained model-->
<!--* Train your own interaction energy or atomic property model-->
<!--To get started, check out the [documentation page](file:///Users/zachglick/gits/AP-Net-mp-temp/docs/build/html/index.html). (TODO: hook up readthedocs)-->
