import os

import torch
from torch import Tensor
from typing import Optional
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from kge.model.distmult import DistMultScorer


class ODistMult(KgeModel):
    r"""Implementation of the DistMult KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=DistMultScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
        print("loading descriptions")
        if self.get_option("mentions_only"):
            self.register_buffer("descriptions", torch.load(os.path.join(self.dataset.folder, "mentions_embs.pt")))
        else:
            self.register_buffer("descriptions", torch.load(os.path.join(self.dataset.folder, "descriptions_embs.pt")))
        print("descriptions loaded")
        self.descriptions.requires_grad = False
        self.projection = torch.nn.Linear(self.get_s_embedder().dim + self.descriptions.shape[1], self.get_s_embedder().dim, bias=False)
        if not init_for_load_only:
            self.get_s_embedder().initialize(self.projection.weight.data)

    def project_with_descriptions(self, indexes, embs):
        descriptions = self.descriptions[indexes.long()]
        embeddings = torch.cat([descriptions, embs], dim=1)
        return self.projection(embeddings)

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None, unseen_mask: Optional[Tensor] = None, ctx=None) -> Tensor:
        if unseen_mask is None or ctx is None:
            s_emb = self.get_s_embedder().embed(s)
        else:
            seen_s = s[~unseen_mask]
            s_emb = torch.empty([len(s), self.get_s_embedder().dim], device=s.device)
            new_s_emb = list()
            for ct in ctx:
                ctx_p_emb = self.get_p_embedder().embed(ct[:, 0].long())
                ctx_o_emb = self.get_o_embedder().embed(ct[:, 1].long())
                new_s_emb.append((ctx_p_emb*ctx_o_emb).mean(dim=0).view(1, -1))
            s_emb[unseen_mask] = torch.cat(new_s_emb, dim=0)
            s_emb[~unseen_mask] = self.get_s_embedder().embed(seen_s)
        p_emb = self.get_p_embedder().embed(p)
        if o is None:
            o_emb = self.get_o_embedder().embed_all()
        else:
            o_emb = self.get_o_embedder().embed(o)

        s_emb = self.project_with_descriptions(s, s_emb)
        o_emb = self.project_with_descriptions(o, o_emb)

        return self._scorer.score_emb(s_emb, p_emb, o_emb, combine="sp_")

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None, unseen_mask: Optional[Tensor] = None, ctx=None) -> Tensor:
        if s is None:
            s_emb = self.get_s_embedder().embed_all()
        else:
            s_emb = self.get_s_embedder().embed(s)
        if unseen_mask is None or ctx is None:
            o_emb = self.get_o_embedder().embed(o)
        else:
            seen_o = o[~unseen_mask]
            o_emb = torch.empty([len(o), self.get_o_embedder().dim], device=o.device)
            o_emb[~unseen_mask] = self.get_o_embedder().embed(seen_o)
            new_o_emb = list()
            for ct in ctx:
                ctx_p_emb = self.get_p_embedder().embed(ct[:, 0].long())
                ctx_s_emb = self.get_o_embedder().embed(ct[:, 1].long())
                new_o_emb.append((ctx_p_emb*ctx_s_emb).mean(dim=0).view(1, -1))
            o_emb[unseen_mask] = torch.cat(new_o_emb, dim=0)
        p_emb = self.get_p_embedder().embed(p)

        s_emb = self.project_with_descriptions(s, s_emb)
        o_emb = self.project_with_descriptions(o, o_emb)

        return self._scorer.score_emb(s_emb, p_emb, o_emb, combine="_po")

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None, unseen_mask: Optional[Tensor] = None, ctx=None) -> Tensor:
        if direction == "o":  # unseen is in s
            o_emb = self.get_o_embedder().embed(o)
            if unseen_mask is None or ctx is None:
                s_emb = self.get_s_embedder().embed(s)
            else:
                seen_s = s[~unseen_mask]
                s_emb = torch.empty([len(s), self.get_s_embedder().dim],
                                    device=s.device)
                new_s_emb = list()
                for ct in ctx:
                    ctx_p_emb = self.get_p_embedder().embed(ct[:, 0].long())
                    ctx_o_emb = self.get_o_embedder().embed(ct[:, 1].long())
                    new_s_emb.append((ctx_p_emb * ctx_o_emb).mean(dim=0).view(1, -1))
                s_emb[unseen_mask] = torch.cat(new_s_emb, dim=0)
                s_emb[~unseen_mask] = self.get_s_embedder().embed(seen_s)
        elif direction == "s":  # unseen is in o
            s_emb = self.get_s_embedder().embed(s)
            if unseen_mask is None or ctx is None:
                o_emb = self.get_o_embedder().embed(o)
            else:
                seen_o = o[~unseen_mask]
                o_emb = torch.empty([len(o), self.get_o_embedder().dim],
                                    device=o.device)
                o_emb[~unseen_mask] = self.get_o_embedder().embed(seen_o)
                new_o_emb = list()
                for ct in ctx:
                    if len(ct) == 0:
                        print("problem")
                        #new_o_emb.append(torch.zeros([1, ]))
                    ctx_p_emb = self.get_p_embedder().embed(ct[:, 0].long())
                    ctx_s_emb = self.get_o_embedder().embed(ct[:, 1].long())
                    new_o_emb.append((ctx_p_emb * ctx_s_emb).mean(dim=0).view(1, -1))
                o_emb[unseen_mask] = torch.cat(new_o_emb, dim=0)
        else:
            raise NotImplementedError()
        p_emb = self.get_p_embedder().embed(p)
        s_emb = self.project_with_descriptions(s, s_emb)
        o_emb = self.project_with_descriptions(o, o_emb)
        return self._scorer.score_emb(s_emb, p_emb, o_emb, combine="spo").view(-1)

    def score_sp_po(
        self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None
    ) -> Tensor:
        sp_scores = self.score_sp(s, p, entity_subset)
        po_scores = self.score_po(p, o, entity_subset)
        return torch.cat([sp_scores, po_scores], dim=1)


