# Copyright (c) Meta Platforms, Inc. and affiliates.

'''
    modified for standard torch
'''
import torch
import torch.nn as nn
from .feature_extractor_modified import FeatureExtractor

class MapEncoder(nn.Module):
    def __init__(self, conf):
        super(MapEncoder, self).__init__()
        self.embeddings = nn.ModuleDict(
            {
                k: nn.Embedding(n + 1, conf['embedding_dim'])
                for k, n in conf['num_classes'].items()
            }
        )
        input_dim = len(conf['num_classes']) * conf['embedding_dim']
        output_dim = conf['output_dim']
        if output_dim is None:
            output_dim = conf['backbone']['output_dim']
        if conf['unary_prior']:
            output_dim += 1
        if conf['backbone'] is None:
            self.encoder = nn.Conv2d(input_dim, output_dim, 1)
        elif conf['backbone'] == "simple":
            self.encoder = nn.Sequential(
                nn.Conv2d(input_dim, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, output_dim, 3, padding=1),
            )
        else:
            self.encoder = FeatureExtractor(
                {
                    **conf['backbone'],
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                }
            )
        self.conf = conf
    def forward(self, data):
        # print(data.shape)
        embeddings = [
            self.embeddings[k](data[:, i])
            for i, k in enumerate(("areas", "ways", "nodes"))
        ]
        embeddings = torch.cat(embeddings, dim=-1).permute(0, 3, 1, 2)
        if isinstance(self.encoder, nn.Module):
            features = self.encoder(embeddings)
        else:
            features = [self.encoder(embeddings)]
        pred = {}
        if self.conf['unary_prior']:
            pred["log_prior"] = [f[:, -1] for f in features]
            features = [f[:, :-1] for f in features]
        pred["map_features"] = features

        # TODO add fc for visual place recognition
        return pred
