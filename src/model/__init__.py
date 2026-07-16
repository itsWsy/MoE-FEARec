"""Model registry.

Some distributed copies of this project contain only a subset of the baseline
model files. Missing optional models must not prevent an available model (for
example FEARec_MI) from starting.
"""

from importlib import import_module


_MODEL_SPECS = {
    "bsarec": ("bsarec", "BSARecModel"),
    "caser": ("caser", "CaserModel"),
    "gru4rec": ("gru4rec", "GRU4RecModel"),
    "sasrec": ("sasrec", "SASRecModel"),
    "bert4rec": ("bert4rec", "BERT4RecModel"),
    "fmlprec": ("fmlprec", "FMLPRecModel"),
    "duorec": ("duorec", "DuoRecModel"),
    "fearec": ("fearec", "FEARecModel"),
    "fearec_mi": ("FEARec_MI", "FEARecMIModel"),
    "moe_fearec": ("FEARec_MI", "FEARecMIModel"),
}


def _build_model_dict():
    model_dict = {}
    for model_name, (module_name, class_name) in _MODEL_SPECS.items():
        qualified_module = "{}.{}".format(__name__, module_name)
        try:
            module = import_module(".{}".format(module_name), package=__name__)
        except ModuleNotFoundError as error:
            # Skip only when the model file itself is absent. If an existing
            # model has a missing dependency, surface that real error instead.
            if error.name == qualified_module:
                continue
            raise
        model_dict[model_name] = getattr(module, class_name)
    return model_dict


MODEL_DICT = _build_model_dict()

