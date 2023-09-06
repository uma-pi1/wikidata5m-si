import math
import time
import sys
import torch
from tqdm import tqdm
from kge.job import EvaluationJob, Job
from kge import Config, Dataset
from collections import defaultdict
import numpy as np
from kge.model import ReciprocalRelationsModel, Hitter
from kge.prepare_few_shot import FewShotSetCreator


class SemiInductiveEntityRankingJob(EvaluationJob):
    """ Entity ranking evaluation protocol """

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        self.config.check(
            "entity_ranking.tie_handling.type",
            ["rounded_mean_rank", "best_rank", "worst_rank"],
        )
        self.tie_handling = self.config.get("entity_ranking.tie_handling.type")

        self.tie_atol = float(self.config.get("entity_ranking.tie_handling.atol"))
        self.tie_rtol = float(self.config.get("entity_ranking.tie_handling.rtol"))

        self.filter_with_test = config.get("entity_ranking.filter_with_test")
        self.filter_splits = self.config.get("entity_ranking.filter_splits")
        if self.eval_split not in self.filter_splits:
            self.filter_splits.append(self.eval_split)

        max_k = min(
            self.dataset.num_entities(),
            max(self.config.get("entity_ranking.hits_at_k_s")),
        )
        self.hits_at_k_s = list(
            filter(lambda x: x <= max_k, self.config.get("entity_ranking.hits_at_k_s"))
        )

        #: Whether to create additional histograms for head and tail slot
        self.head_and_tail = config.get("entity_ranking.metrics_per.head_and_tail")

        if type(self.model) is ReciprocalRelationsModel and type(self.model._base_model) is Hitter:
            self.neighborhood_size = self.config.get(
                "reciprocal_relations_model.base_model.neighborhood_size"
            )

        if "cuda" in self.device:
            torch.backends.cuda.matmul.allow_tf32 = False

        self.batch_size = 1
        self.few_shot_creator = FewShotSetCreator(
            self.config.get("dataset.name"),
            split=self.config.get("eval.split"),
            use_inverse=True,
            context_selection=self.config.get("semi_inductive_entity_ranking.context_selection")
        )

        if self.__class__ == SemiInductiveEntityRankingJob:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        super()._prepare()
        """Construct all indexes needed to run."""

        # create data and precompute indexes
        self.triples = self.dataset.split(self.config.get("eval.split"))
        self.few_shot_dataset = self.few_shot_creator.create_few_shot_dataset(
            num_shots=self.config.get("semi_inductive_entity_ranking.num_shots")
        )
        for split in self.filter_splits:
            self.dataset.index(f"{split}_sp_to_o")
            self.dataset.index(f"{split}_po_to_s")
        if "test" not in self.filter_splits and self.filter_with_test:
            self.dataset.index("test_sp_to_o")
            self.dataset.index("test_po_to_s")

        # and data loader
        self.loader = torch.utils.data.DataLoader(
            self.few_shot_dataset,
            collate_fn=self._collate,
            shuffle=False,
            batch_size=1,  # we enforce this for a simpler implementation as we only score in one direction
            num_workers=self.config.get("eval.num_workers"),
            pin_memory=self.config.get("eval.pin_memory"),
        )

    def _collate(self, batch):
        batch = batch[0]
        #batch = self.few_shot_dataset[batch]
        # bringing context in the right format for hitter [bs, nu
        context = torch.from_numpy(batch["context"])[:, 3:].fliplr().unsqueeze(0)
        # context = torch.flip(context.view(-1, 3)[:, 1:].T, dims=[0]).unsqueeze(0)
        batch["context"] = context
        batch["triple"] = torch.from_numpy(batch["triple"])
        return batch

    @torch.no_grad()
    def _evaluate(self):
        # create initial trace entry
        self.current_trace["epoch"] = dict(
            type="semi_inductive_entity_ranking",
            scope="epoch",
            split=self.eval_split,
            epoch=self.epoch,
            batches=len(self.loader),
            size=len(self.triples),
        )

        # run pre-epoch hooks (may modify trace)
        for f in self.pre_epoch_hooks:
            f(self)

        # let's go
        epoch_time = -time.time()
        ranks = list()
        head_ranks = list()
        tail_ranks = list()
        for batch_number, batch in enumerate(tqdm(self.loader)):
            context = batch["context"].to(self.device)
            triple = batch["triple"].to(self.device)
            scores = self.model.score_sp_semi_ind(
                s=triple[0].unsqueeze(0).view(1, 1),
                p=triple[1].unsqueeze(0).view(1, 1),
                ctx_ids=context,
                ctx_size=torch.IntTensor([context.shape[1]]).to(self.device)
            )
            true_score = scores[0, triple[2]]
            num_higher = torch.sum(scores > true_score)
            num_ties = torch.sum(scores == true_score)
            rank = num_higher + num_ties // 2 + 1
            ranks.append(rank.item())
            if triple[1] >= self.dataset.num_relations():
                head_ranks.append(rank.item())
            else:
                tail_ranks.append(rank.item())
        ranks = np.array(ranks)
        mrr = np.mean(1/ranks).item()
        h1 = np.mean(ranks <= 1).item()
        h3 = np.mean(ranks <= 3).item()
        h5 = np.mean(ranks <= 5).item()
        h10 = np.mean(ranks <= 10).item()
        h100 = np.mean(ranks <= 100).item()

        epoch_time += time.time()
        output = dict(
            event="eval_completed",
            num_shots=self.config.get("semi_inductive_entity_ranking.num_shots")
        )

        for suffix, rs in zip(["", "_head", "_tail"], [ranks, head_ranks, tail_ranks]):
            rs = np.array(rs)
            mrr = np.mean(1/rs).item()
            h1 = np.mean(rs <= 1).item()
            h3 = np.mean(rs <= 3).item()
            h5 = np.mean(rs <= 5).item()
            h10 = np.mean(rs <= 10).item()
            h100 = np.mean(rs <= 100).item()
            o = {
                f"mean_reciprocal_rank{suffix}": mrr,
                f"hits_at_1{suffix}": h1,
                f"hits_at_3{suffix}": h3,
                f"hits_at_5{suffix}": h5,
                f"hits_at_10{suffix}": h10,
                f"hits_at_100{suffix}": h100,
            }
            output.update(o)

        self.current_trace["epoch"].update(
            output
        )
        print(output)
