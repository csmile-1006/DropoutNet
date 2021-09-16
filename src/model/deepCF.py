import torch
from torch import nn


class DeepCF(nn.Module):
    def __init__(
        self, latent_dim, user_content_dim, item_content_dim, model_select, output_dim
    ):
        super(DeepCF, self).__init__()
        for tp in ["user", "item"]:
            dims = [latent_dim + eval(f"{tp}_content_dim")] + model_select
            layers = []
            prev_dim = dims[0]
            for dim in dims[1:]:
                layers.append(nn.Linear(prev_dim, dim))
                layers.append(nn.BatchNorm1d(dim))
                layers.append(nn.Tanh())
                prev_dim = dim
            setattr(self, f"{tp}_transform", nn.Sequential(*layers))

        self.user_last_linear = nn.Linear(model_select[-1], output_dim)
        self.item_last_linear = nn.Linear(model_select[-1], output_dim)
        self.initialize()

    def truncated_normal(self, t, mean=0.0, std=0.01):
        nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(
                cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t
            )
        return t

    def initialize(self):
        for tp in ["user", "item"]:
            transform = getattr(self, f"{tp}_transform")
            for layer in transform:
                if isinstance(layer, (nn.Tanh, nn.BatchNorm1d)):
                    continue
                layer.weight.data = self.truncated_normal(layer.weight.data)
                nn.init.zeros_(layer.bias)
            last_linear = getattr(self, f"{tp}_last_linear")
            last_linear.weight.data = self.truncated_normal(last_linear.weight.data)
            nn.init.zeros_(last_linear.bias)

    def forward(self, user_latent, item_latent, user_content, item_content):
        user_input = torch.cat([user_latent, user_content], 1)
        item_input = torch.cat([item_latent, item_content], 1)
        user_last = self.user_transform(user_input)
        item_last = self.item_transform(item_input)

        user_embedding = self.user_last_linear(user_last)
        item_embedding = self.item_last_linear(item_last)

        return user_embedding, item_embedding
