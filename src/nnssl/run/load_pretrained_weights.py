import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP


def load_pretrained_weights(network, fname, verbose=False):
    saved_model = torch.load(fname, weights_only=True)
    pretrained_dict = saved_model["network_weights"]

    skip_strings_in_pretrained = [".seg_layers."]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()

    def _adapt_to_target(pretrained, target):
        """Copy pretrained into the matching corner of a (same- or larger-shaped)
        target tensor, zero-initializing any new entries. Returns None if the
        target is smaller in any dim (genuinely incompatible)."""
        if pretrained.dim() != target.dim():
            return None
        if any(p > t for p, t in zip(pretrained.shape, target.shape)):
            return None
        new_w = torch.zeros_like(target)
        slices = tuple(slice(0, p) for p in pretrained.shape)
        new_w[slices] = pretrained.to(new_w)
        return new_w

    adapted_dict = {}
    for k, v in pretrained_dict.items():
        if k not in model_dict or any(s in k for s in skip_strings_in_pretrained):
            continue
        target = model_dict[k]
        if v.shape == target.shape:
            adapted_dict[k] = v
        else:
            nw = _adapt_to_target(v, target)
            if nw is not None:
                adapted_dict[k] = nw
                print(
                    f"Adapted '{k}': {tuple(v.shape)} -> {tuple(target.shape)} "
                    f"(pretrained copied, new entries zero-initialized)"
                )
            else:
                print(
                    f"Skipping '{k}': incompatible shape "
                    f"{tuple(v.shape)} vs {tuple(target.shape)}"
                )

    model_dict.update(adapted_dict)

    print(
        "################### Loading pretrained weights from file ",
        fname,
        "###################",
    )
    if verbose:
        print("Overlapping / adapted blocks:")
        for key, value in adapted_dict.items():
            print(key, "shape", tuple(value.shape))
        print("################### Done ###################")

    mod.load_state_dict(model_dict)
