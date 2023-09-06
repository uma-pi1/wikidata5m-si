import time

import torch
import torch.utils.data

from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn
from kge.model import ReciprocalRelationsModel, Hitter
import numpy as np


class TrainingJob1vsSample(TrainingJob):
    """Samples SPO pairs and queries sp_ and _po, treating all other entities as negative."""

    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        config.log("Initializing spo training job...")
        self.type_str = "1vsSample"
        self.s_num_neg = self.config.get("1vsSample.num_samples.s")
        self.o_num_neg = self.config.get("1vsSample.num_samples.o")
        self.max_train_entity = self.dataset.split("train")[:, [0, 2]].max()
        if type(self.model) is ReciprocalRelationsModel and type(self.model._base_model) is Hitter:
            self.neighborhood_size = self.config.get(
                "reciprocal_relations_model.base_model.neighborhood_size"
            )

        if self.__class__ == TrainingJob1vsSample:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()

        self.num_examples = self.dataset.split(self.train_split).size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            #collate_fn=lambda batch: {
            #    "triples": self.dataset.split(self.train_split)[batch, :].long()
            #},
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _get_collate_fun(self):
        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a tuple of:

            - triples (tensor of shape [n,3], ),
            - negative_samples (list of tensors of shape [n,num_samples]; 3 elements
              in order S,P,O)
            """

            triples = self.dataset.split(self.train_split)[batch, :].long()
            all_ctx_s = np.empty([len(batch), self.neighborhood_size, 2], dtype=int)
            all_ctx_o = np.empty([len(batch), self.neighborhood_size, 2], dtype=int)
            ctx_size_s = torch.empty([len(batch)], dtype=torch.int)
            ctx_size_o = torch.empty([len(batch)], dtype=torch.int)
            for i, triple in enumerate(triples):
                ctx_s = self.dataset.index("1hop").get(triple[0])
                ctx_o = self.dataset.index("1hop").get(triple[2])
                all_ctx_s[i, :len(ctx_s), 0] = ctx_s[:, 1]
                all_ctx_s[i, :len(ctx_s), 1] = ctx_s[:, 0]
                all_ctx_o[i, :len(ctx_o), 0] = ctx_o[:, 1]
                all_ctx_o[i, :len(ctx_o), 1] = ctx_o[:, 0]
                ctx_size_s[i] = len(ctx_s)
                ctx_size_o[i] = len(ctx_o)
            #all_ctx_s = torch.stack(all_ctx_s)
            #all_ctx_o = torch.stack(all_ctx_o)
            return {
                "triples": triples,
                "ctx_s": torch.from_numpy(all_ctx_s),
                "ctx_o": torch.from_numpy(all_ctx_o),
                "ctx_size_s": ctx_size_s,
                "ctx_size_o": ctx_size_o
            }

        return collate

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        result.size = len(batch["triples"])

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: TrainingJob._ProcessBatchResult,
    ):
        batch_size = result.size

        # prepare
        result.prepare_time -= time.time()
        triples = batch["triples"][subbatch_slice].to(self.device)
        ctx_s = batch["ctx_s"][subbatch_slice].to(self.device)
        ctx_o = batch["ctx_o"][subbatch_slice].to(self.device)
        ctx_size_s = batch["ctx_size_s"][subbatch_slice].to(self.device)
        ctx_size_o = batch["ctx_size_o"][subbatch_slice].to(self.device)
        result.prepare_time += time.time()
        s_shared_neg = torch.randint(self.max_train_entity, [self.s_num_neg], device=triples.device)
        o_shared_neg = torch.randint(self.max_train_entity, [self.o_num_neg], device=triples.device)
        # this is an inefficient hack to be able to provide ground truth for self
        # mlm loss easily
        s_target = torch.cat((triples[:, 0].view(-1), s_shared_neg.view(-1)),)
        o_target = torch.cat((triples[:, 2].view(-1), o_shared_neg.view(-1)),)

        gt_ids = torch.arange(len(triples), device=triples.device)

        # forward/backward pass (sp)
        result.forward_time -= time.time()
        scores_sp = self.model.score_sp(
            triples[:, 0],
            triples[:, 1],
            o=o_target,
            #ground_truth=gt_ids
            ground_truth=triples[:, 2],
            ctx_ids=ctx_s,
            ctx_size=ctx_size_s,
        )
        if isinstance(scores_sp, tuple):
            scores_sp, self_pred_loss_sp = scores_sp
            #loss_value_sp = self.loss(scores_sp, triples[:, 2]) / batch_size
            loss_value_sp = self.loss(scores_sp, gt_ids) / batch_size
            loss_value_sp = loss_value_sp + self_pred_loss_sp
            result.avg_loss_self += self_pred_loss_sp.item()
        else:
            #loss_value_sp = self.loss(scores_sp, triples[:, 2]) / batch_size
            loss_value_sp = self.loss(scores_sp, gt_ids) / batch_size
        result.avg_loss += loss_value_sp.item()
        result.forward_time += time.time()
        result.backward_time = -time.time()
        if not self.is_forward_only:
            loss_value_sp.backward()
        result.backward_time += time.time()

        # forward/backward pass (po)
        result.forward_time -= time.time()
        scores_po = self.model.score_po(
            triples[:, 1],
            triples[:, 2],
            s=s_target,
            #ground_truth=gt_ids
            ground_truth=triples[:, 0],
            ctx_ids=ctx_o,
            ctx_size=ctx_size_o,
        )
        if isinstance(scores_po, tuple):
            scores_po, self_pred_loss_po = scores_po
            #loss_value_po = self.loss(scores_po, triples[:, 0]) / batch_size
            loss_value_po = self.loss(scores_po, gt_ids) / batch_size
            loss_value_po = loss_value_po + self_pred_loss_po
            result.avg_loss_self += self_pred_loss_po.item()
        else:
            #loss_value_po = self.loss(scores_po, triples[:, 0]) / batch_size
            loss_value_po = self.loss(scores_po, gt_ids) / batch_size
        result.avg_loss += loss_value_po.item()
        result.forward_time += time.time()
        result.backward_time -= time.time()
        if not self.is_forward_only:
            loss_value_po.backward()
        result.backward_time += time.time()
