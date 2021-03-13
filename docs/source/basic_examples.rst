Basic Examples
==============

The following code uses an ensemble of pretrained AP-Net to predict the interaction energy of a water dimer:

.. code-block::

    import apnet
    import qcelemental as qcel

    dimer = qcel.models.Molecule.from_data("""
    0 1
    Ne 0.0 0.0 0.0
    --
    0 1
    Ne 1.0 1.0 1.0
    """)

    prediction, uncertainty = apnet.predict_sapt(dimer)

Note that ``apnet`` uses the :class:`qcelemental.models.Molecule` class to represent molecular monomers and dimers.
More information about how to instantiate a ``Molecule`` can be found at the `QCElemental docs <https://qcelemental.readthedocs.io/en/latest/model_molecule.html#creation>`__.
See :ref:`advanced_examples` for additional ``apnet`` examples.
