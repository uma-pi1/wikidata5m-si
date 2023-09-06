from kge import Config, Configurable
import torch.optim
from torch.optim.lr_scheduler import _LRScheduler
import re
from operator import or_
from functools import reduce
from .bert_optim import Adamax


class KgeOptimizer:
    """ Wraps torch optimizers """

    @staticmethod
    def create(config, model):
        """ Factory method for optimizer creation """
        try:
            name = config.get("train.optimizer.default.type")
            if name == "Adamax":
                optimizer = Adamax
            else:
                optimizer = getattr(torch.optim, name)
            #if name == "Adamax":
            #    optimizer = Adamax
            return optimizer(
                KgeOptimizer._get_parameters_and_optimizer_args(config, model),
                **config.get("train.optimizer.default.args"),
            )
        except AttributeError:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError(
                f"Could not create optimizer {config.get('train.optimizer')}. "
                f"Please specify an optimizer provided in torch.optim"
            )

    @staticmethod
    def _get_parameters_and_optimizer_args(config, model):
        """
        Group named parameters by regex strings provided with optimizer args.
        Constructs a list of dictionaries of the form:
        [
            {
                "name": name of parameter group
                "params": list of parameters to optimize
                # parameter specific options as for example learning rate
                ...
            },
            ...
        ]
        """

        named_parameters = dict(model.named_parameters())
        optimizer_settings = config.get("train.optimizer")
        parameter_names_per_search = dict()
        # filter named parameters by regex string
        for group_name, parameter_group in optimizer_settings.items():
            if group_name == "default":
                continue
            if "type" in parameter_group.keys():
                raise NotImplementedError("Multiple optimizer types are not yet supported.")
            search_pattern = re.compile(parameter_group["regex"])
            filtered_named_parameters = set(
                filter(search_pattern.match, named_parameters.keys())
            )
            parameter_names_per_search[group_name] = filtered_named_parameters

        # check if something was matched by multiple strings
        parameter_values = list(parameter_names_per_search.values())
        for i, (group_name, param) in enumerate(parameter_names_per_search.items()):
            for j in range(i + 1, len(parameter_names_per_search)):
                intersection = set.intersection(param, parameter_values[j])
                if len(intersection) > 0:
                    raise ValueError(
                        f"The parameters {intersection}, were matched by the optimizer "
                        f"group {group_name} and {list(parameter_names_per_search.keys())[j]}"
                    )
        resulting_parameters = []
        for group_name, params in parameter_names_per_search.items():
            optimizer_settings[group_name]["args"]["params"] = [
                named_parameters[param] for param in params
            ]
            optimizer_settings[group_name]["args"]["name"] = group_name
            resulting_parameters.append(optimizer_settings[group_name]["args"])

        # add unmatched parameters to default group
        if len(parameter_names_per_search) > 0:
            default_parameter_names = set.difference(
                set(named_parameters.keys()),
                reduce(or_, list(parameter_names_per_search.values())),
            )
            default_parameters = [
                named_parameters[default_parameter_name]
                for default_parameter_name in default_parameter_names
            ]
            resulting_parameters.append(
                {"params": default_parameters, "name": "default"}
            )
        else:
            # no parameters matched, add everything to default group
            resulting_parameters.append(
                {"params": list(model.parameters()), "name": "default"}
            )
        return resulting_parameters


class KgeLRScheduler(Configurable):
    """ Wraps torch learning rate (LR) schedulers """

    def __init__(self, config: Config, optimizer):
        super().__init__(config)
        name = config.get("train.lr_scheduler")
        args = config.get("train.lr_scheduler_args")
        self._lr_scheduler: _LRScheduler = None
        if name != "":
            # check for consistency of metric-based scheduler
            self._metric_based = name in ["ReduceLROnPlateau"]
            if self._metric_based:
                desired_mode = "max" if config.get("valid.metric_max") else "min"
                if "mode" in args:
                    if args["mode"] != desired_mode:
                        raise ValueError(
                            (
                                "valid.metric_max ({}) and train.lr_scheduler_args.mode "
                                "({}) are inconsistent."
                            ).format(config.get("valid.metric_max"), args["mode"])
                        )
                    # all fine
                else:  # mode not set, so set it
                    args["mode"] = desired_mode
                    config.set("train.lr_scheduler_args.mode", desired_mode, log=True)

            if name == "LinearLR":
                try:
                    self._lr_scheduler = LinearLR(optimizer, **args)
                    return
                except Exception as e:
                    raise ValueError(
                        (
                            "Invalid LR scheduler {} or scheduler arguments {}. "
                            "Error: {}"
                        ).format(name, args, e)
                    )

            # create the scheduler
            try:
                self._lr_scheduler = getattr(torch.optim.lr_scheduler, name)(
                    optimizer, **args
                )
            except Exception as e:
                raise ValueError(
                    (
                        "Invalid LR scheduler {} or scheduler arguments {}. "
                        "Error: {}"
                    ).format(name, args, e)
                )

    def step(self, metric=None):
        if self._lr_scheduler is None:
            return
        if self._metric_based:
            if metric is not None:
                # metric is set only after validation has been performed, so here we
                # step
                self._lr_scheduler.step(metrics=metric)
        else:
            # otherwise, step after every epoch
            self._lr_scheduler.step()

    def state_dict(self):
        if self._lr_scheduler is None:
            return dict()
        else:
            return self._lr_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        if self._lr_scheduler is None:
            pass
        else:
            self._lr_scheduler.load_state_dict(state_dict)


class LinearLR(_LRScheduler):
    """
    copied from pytorch 1.10
    Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.03125  if epoch == 1
        >>> # lr = 0.0375   if epoch == 2
        >>> # lr = 0.04375  if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = LinearLR(self.opt, start_factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1,
                 verbose=False):
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError('Starting multiplicative factor expected to be between 0 and 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super(LinearLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] * self.start_factor for group in self.optimizer.param_groups]

        if (self.last_epoch > self.total_iters):
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['lr'] * (1. + (self.end_factor - self.start_factor) /
                (self.total_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor)))
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * (self.start_factor +
                (self.end_factor - self.start_factor) * min(self.total_iters, self.last_epoch) / self.total_iters)
                for base_lr in self.base_lrs]

