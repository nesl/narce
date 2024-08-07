import torch
import torch.nn as nn
from mamba_ssm.models.config_mamba import MambaConfig
from models import MambaModel


class NARMamba(nn.Module):
    def __init__(
    self, 
    mamba_config, 
    nar_vocab_size,
    out_cls_dim,
    ) -> None:
        super().__init__()
        d_model = mamba_config.d_model
        self.embedding = nn.Embedding(nar_vocab_size, d_model)

        nar_backbone = MambaModel(mamba_config)
        classifier_head = nn.Linear(d_model, out_cls_dim)
        self.nar = nn.Sequential(nar_backbone, classifier_head)

    def forward(self, x):
        x = self.embedding(x)
        x = self.nar(x)
        return x


class AdapterMamba(nn.Module):
    def __init__(
    self,
    d_model,
    n_layer,
    ) -> None:
        super().__init__()
        mamba_config = MambaConfig(d_model=d_model, n_layer=n_layer, ssm_cfg={"layer": "Mamba2", "headdim": 32,})
        self.model = MambaModel(mamba_config)

    def forward(self, x):
        x = self.model(x)
        return x


class NarcePipeline(nn.Module):
    def __init__(
    self, 
    frozen_nar,
    adapter_model,
    ) -> None:
        super().__init__()
        self.adapter = adapter_model
        self.nar = frozen_nar

    def forward(self, x):
        x = self.adapter(x)
        # Map into the latent space of NAR
        x = self.nar(x)
        return x


if __name__=="__main__":
    batch, length, dim = 2, 60, 128
    x = torch.randn(batch, length, dim).to('cuda')

    mamba_config = MambaConfig(d_model=128, n_layer=12, ssm_cfg={"layer": "Mamba2", "headdim": 32,})
    model = MambaModel(mamba_config).to('cuda')

    y = model(x)
    print(x.shape, y.shape)
    # assert y.shape == x.shape

 