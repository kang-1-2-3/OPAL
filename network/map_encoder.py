import torch
import torch.nn as nn


class MapEncoder(nn.Module):
    def __init__(self, conf):
        super(MapEncoder, self).__init__()
        self.embeddings = nn.ModuleDict(
            {
                k: nn.Embedding(n + 1, conf['embedding_dim'])
                for k, n in conf['num_classes'].items()
            }
        )

    def forward(self, data):
        # print(data.shape)
        embeddings = [
            self.embeddings[k](data[:, i])
            for i, k in enumerate(("areas", "ways", "nodes"))
        ]
        embeddings = torch.cat(embeddings, dim=-1).permute(0, 3, 1, 2) # [2, 48, 200, 200]
        
        return embeddings
