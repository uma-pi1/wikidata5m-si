import torch
import time

from kge import Config, Dataset
from kge.job import Job, TrainingOrEvaluationJob, TrainingJob
from kge.model import KgeModel
from kge.job.trace import format_trace_entry
from kge.misc import init_from
from kge.indexing import add_index_function

from typing import Dict, Optional, Any


class EvaluationJob(TrainingOrEvaluationJob):
    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job)

        self.config = config
        self.dataset = dataset
        self.model = model
        self.batch_size = config.get("eval.batch_size")
        self.device = self.config.get("job.device")
        self.config.check("train.trace_level", ["example", "batch", "epoch"])
        self.trace_examples = self.config.get("eval.trace_level") == "example"
        self.trace_batch = (
            self.trace_examples or self.config.get("train.trace_level") == "batch"
        )
        self.eval_split = self.config.get("eval.split")
        self.epoch = -1
        self.train_few_shot = self.config.get("eval.few_shot")

        # all done, run job_created_hooks if necessary
        if self.__class__ == EvaluationJob:
            for f in Job.job_created_hooks:
                f(self)

    @staticmethod
    def create(config, dataset, parent_job=None, model=None):
        """Factory method to create an evaluation job """

        eval_type = config.get("eval.type")
        class_name = config.get_default(f"{eval_type}.class_name")
        return init_from(
            class_name,
            config.modules(),
            config,
            dataset,
            parent_job=parent_job,
            model=model,
        )

    def _prepare(self):
        """Prepare this job for running.
        Guaranteed to be called exactly once before running the first epoch.

        """
        super()._prepare()
        self.model.prepare_job(self)  # let the model add some hooks

    def _run(self) -> Dict[str, Any]:
        was_training = self.model.training
        self.model.eval()
        self.config.log(
            "Evaluating on "
            + self.eval_split
            + " data (epoch {})...".format(self.epoch)
        )

        #if self.train_few_shot:
        #    self._train_few_shot()
        self._evaluate()

        # if validation metric is not present, try to compute it
        metric_name = self.config.get("valid.metric")
        if metric_name not in self.current_trace["epoch"]:
            self.current_trace["epoch"][metric_name] = eval(
                self.config.get("valid.metric_expr"),
                None,
                dict(config=self.config, **self.current_trace["epoch"]),
            )

        # run hooks (may modify trace)
        for f in self.post_epoch_hooks:
            f(self)

        # output the trace, then clear it
        trace_entry = self.trace(**self.current_trace["epoch"], echo=False, log=True)
        self.config.log(
            format_trace_entry("eval_epoch", trace_entry, self.config), prefix="  "
        )
        self.current_trace["epoch"] = None

        # reset model and return metrics
        if was_training:
            self.model.train()

        self.config.log("Finished evaluating on " + self.eval_split + " split.")

        return trace_entry

    def _evaluate(self):
        """
        Compute evaluation metrics, output results to trace file.
        The results of the evaluation must be written into self.current_trace["epoch"]
        """
        raise NotImplementedError

    def _load(self, checkpoint: Dict):
        if checkpoint["type"] not in ["train", "package"]:
            raise ValueError("Can only evaluate train and package checkpoints.")
        self.resumed_from_job_id = checkpoint.get("job_id")
        self.epoch = checkpoint["epoch"]
        self.trace(
            event="job_resumed", epoch=self.epoch, checkpoint_file=checkpoint["file"]
        )

    @classmethod
    def create_from(
        cls,
        checkpoint: Dict,
        new_config: Config = None,
        dataset: Dataset = None,
        parent_job=None,
        eval_split: Optional[str] = None,
    ) -> Job:
        """
        Creates a Job based on a checkpoint
        Args:
            checkpoint: loaded checkpoint
            new_config: optional config object - overwrites options of config
                              stored in checkpoint
            dataset: dataset object
            parent_job: parent job (e.g. search job)
            eval_split: 'valid' or 'test'.
                        Defines the split to evaluate on.
                        Overwrites split defined in new_config or config of
                        checkpoint.

        Returns: Evaluation-Job based on checkpoint

        """
        if new_config is None:
            new_config = Config(load_default=False)
        if not new_config.exists("job.type") or new_config.get("job.type") != "eval":
            new_config.set("job.type", "eval", create=True)
        if eval_split is not None:
            new_config.set("eval.split", eval_split, create=True)

        return super().create_from(checkpoint, new_config, dataset, parent_job)

    def _freeze_and_store_paramters(self):
        # freeze entity parameters
        # (on relation we wont have changes anyways except for few shot)
        from kge.model import HashLookupEmbedder
        if type(self.model.get_s_embedder()) is HashLookupEmbedder:
            self.model.get_s_embedder().partition_embedder._embeddings.weight.requires_grad = False
            self.model.get_s_embedder().base_embedder._embeddings.weight.requires_grad = False
        else:
            self.model.get_s_embedder()._embeddings.weight.requires_grad = False

        # store previous relation parameters
        from copy import deepcopy
        self.previous_relation_parameters = []
        if type(self.model.get_p_embedder()) is HashLookupEmbedder:
            self.model.get_p_embedder().partition_embedder._embeddings.weight.requires_grad = False
            self.previous_relation_parameters.append(deepcopy(self.model.get_p_embedder().partition_embedder._embeddings.weight))
            self.previous_relation_parameters.append(deepcopy(self.model.get_p_embedder().base_embedder._embeddings.weight))
        else:
            self.previous_relation_parameters.append(deepcopy(self.model.get_p_embedder()._embeddings.weight))

        # store previous entity parameters
        self.previous_entity_parameters = []
        if type(self.model.get_s_embedder()) is HashLookupEmbedder:
            self.previous_entity_parameters.append(deepcopy(self.model.get_s_embedder().partition_embedder._embeddings.weight))
            self.previous_entity_parameters.append(deepcopy(self.model.get_s_embedder().base_embedder._embeddings.weight))
        else:
            self.previous_entity_parameters.append(deepcopy(self.model.get_s_embedder()._embeddings.weight))

    def _unfreeze_and_reset_parameters(self):
        from kge.model import HashLookupEmbedder
        if type(self.model.get_s_embedder()) is HashLookupEmbedder:
            self.model.get_s_embedder().partition_embedder._embeddings.weight.requires_grad = True
            self.model.get_s_embedder().base_embedder._embeddings.weight.requires_grad = True
        else:
            self.model.get_s_embedder()._embeddings.weight.requires_grad = True

        if type(self.model.get_p_embedder()) is HashLookupEmbedder:
            self.model.get_p_embedder().partition_embedder._embeddings.weight.requires_grad = True
            self.model.get_p_embedder().partition_embedder._embeddings.weight[:, :] = self.previous_relation_parameters[0]
            self.model.get_p_embedder().base_embedder._embeddings.weight[:, :] = self.previous_relation_parameters[1]
        else:
            self.model.get_p_embedder()._embeddings.weight[:, :] = self.previous_relation_parameters[0]

        if type(self.model.get_s_embedder()) is HashLookupEmbedder:
            self.model.get_s_embedder().partition_embedder._embeddings.weight.requires_grad = True
            self.model.get_s_embedder().partition_embedder._embeddings.weight[:, :] = self.previous_entity_parameters[0]
            self.model.get_s_embedder().base_embedder._embeddings.weight[:, :] = self.previous_entity_parameters[1]
        else:
            self.model.get_s_embedder()._embeddings.weight[:, :] = self.previous_entity_parameters[0]

    def _prep_few_shot_dataset(self, num_few_shots):
        unique_relations, rel_counts = torch.unique(self.triples[:, 1], return_counts=True, sorted=True)
        triples = []
        few_shots = []
        beginning = 0
        for i, rel_ind in enumerate(rel_counts):
            end = beginning+rel_ind
            few_shots.append(self.triples[beginning:beginning+num_few_shots])
            triples.append(self.triples[beginning+num_few_shots:end])
            beginning += rel_ind
        #triples.append(self.triples[end:])
        self.triples = torch.cat(triples)
        self.few_shot_triples = torch.cat(few_shots)
        self.dataset._triples["train_few_shot"] = self.few_shot_triples
        self.filter_splits.append("train_few_shot")
        add_index_function(self.dataset, "train_few_shot")

    def _train_few_shot(self):
        self._freeze_and_store_paramters()
        # split triples to few shot
        # split by relationship type
        few_shot_config = self.model.config.clone()
        import tempfile
        few_shot_config.log_folder = tempfile.mkdtemp(prefix="kge-")
        few_shot_config.folder = few_shot_config.log_folder
        few_shot_config.set("train.max_epochs", self.config.get("eval.few_shot_args.num_epochs"))
        few_shot_config.set("train.checkpoint.keep", 0)
        few_shot_config.set("train.checkpoint.every", 0)
        few_shot_config.set("valid.every", 0)
        few_shot_config.set("job.type", "train")
        few_shot_config.set(
            "train.optimizer.default.args.lr",
            few_shot_config.get("train.optimizer.default.args.lr")/self.config.get("eval.few_shot_args.lr_reduction_factor")
        )
        few_shot_config.load_options(self.config.get("eval.few_shot_params"))

        few_shot_train_job = Job.create(few_shot_config, model=self.model)
        few_shot_train_job.dataset._triples[self.config.get("train.split")] = self.few_shot_triples
        few_shot_train_job.is_few_shot_only = True
        few_shot_train_job.run()
        self.model.eval()

