import os
import torch.nn
import torch.nn.functional
from kge.model import KgeEmbedder, LookupEmbedder


class DescriptionEmbedder(LookupEmbedder):
    """Adds a linear projection layer to a base embedder."""

    def __init__(
        self, config, dataset, configuration_key, vocab_size, init_for_load_only=False
    ):
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only, vocab_size=vocab_size
        )

        # initialize projection
        if self.dim < 0:
            self.dim = self.base_embedder.dim
        #self.dropout = self.get_option("dropout")
        self.regularize = self.check_option("regularize", ["", "lp"])
        print("loading descriptions")
        if self.get_option("mentions_only"):
            self.register_buffer("descriptions", torch.load(os.path.join(self.dataset.folder, "mentions_embs.pt")))
        else:
            self.register_buffer("descriptions", torch.load(os.path.join(self.dataset.folder, "descriptions_embs.pt")))
        print("descriptions loaded")
        self.use_bias = self.get_option("use_bias")
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.rand(1, self.dim))
        self.descriptions.requires_grad = False
        self.projection = torch.nn.Linear(self.dim + self.descriptions.shape[1], self.dim, bias=False)
        if not init_for_load_only:
            self.initialize(self.projection.weight.data)

    def _embed(self, embeddings):
        embeddings = self.projection(embeddings)
        #if self.dropout > 0:
        #    embeddings = torch.nn.functional.dropout(
        #        embeddings, p=self.dropout, training=self.training
        #    )
        return embeddings

    def embed(self, indexes):
        descriptions = self.descriptions[indexes.long()]
        base_embs = super().embed(indexes)
        if self.use_bias:
            base_embs += self.bias
        embeddings = torch.cat([descriptions, base_embs], dim=1)
        return self._embed(embeddings)

    def embed_all(self):
        base_embs = super().embed_all()
        if self.use_bias:
            base_embs += self.bias
        embeddings = torch.cat([self.descriptions, base_embs], dim=1)
        return self._embed(embeddings)

