"""Model definitions for the molecular force field library."""

try:
    from molecular_force_field.models.e3nn_layers import (
        E3Conv,
        E3Conv2,
        E3_TransformerLayer_multi,
    )
    from molecular_force_field.models.mlp import MainNet, MainNet2
    from molecular_force_field.models.losses import RMSELoss
    from molecular_force_field.models.cartesian_e3_layers import (
        CartesianTransformerLayer,
        CartesianTransformerLayerStrict,
        CartesianTransformerLayerLoose,
        CartesianE3ConvSparse,
        CartesianE3Conv2Sparse,
        CartesianE3ConvStrict,
        CartesianE3Conv2Strict,
        CartesianE3ConvSparseLoose,
        CartesianE3Conv2SparseLoose,
        CartesianFullyConnectedTensorProduct,
        CartesianFullTensorProduct,
        CartesianProductStrict,
        EquivariantTensorProduct,
    )
    
    __all__ = [
        "E3Conv",
        "E3Conv2",
        "E3_TransformerLayer_multi",
        "CartesianTransformerLayer",
        "CartesianTransformerLayerStrict",
        "CartesianTransformerLayerLoose",
        "CartesianE3ConvSparse",
        "CartesianE3Conv2Sparse",
        "CartesianE3ConvStrict",
        "CartesianE3Conv2Strict",
        "CartesianE3ConvSparseLoose",
        "CartesianE3Conv2SparseLoose",
        "CartesianFullyConnectedTensorProduct",
        "CartesianFullTensorProduct",
        "CartesianProductStrict",
        "EquivariantTensorProduct",
        "MainNet",
        "MainNet2",
        "RMSELoss",
    ]
except ImportError:
    __all__ = []