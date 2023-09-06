
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
        p = self.get_p_embedder().embed(p)
        if o is None:
            o = self.get_o_embedder().embed_all()
        else:
            o = self.get_o_embedder().embed(o)

        return self._scorer.score_emb(s_emb, p, o, combine="sp_")

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None, unseen_mask: Optional[Tensor] = None, ctx=None) -> Tensor:
        if s is None:
            s = self.get_s_embedder().embed_all()
        else:
            s = self.get_s_embedder().embed(s)
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
        p = self.get_p_embedder().embed(p)

        return self._scorer.score_emb(s, p, o_emb, combine="_po")

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
        p = self.get_p_embedder().embed(p)
        return self._scorer.score_emb(s_emb, p, o_emb, combine="spo").view(-1)


