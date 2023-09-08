import os
import json
import torch

from config import args
from predict import BertPredictor
from dict_hub import get_entity_dict
from evaluate import eval_single_direction
from logger_config import logger

assert args.task == 'wiki5m_trans', 'This script is only used for wiki5m transductive setting'

entity_dict = get_entity_dict()
SHARD_SIZE = 1000000


def eval_single_direction2(predictor: BertPredictor,
                          entity_tensor: torch.tensor,
                          eval_forward=True,
                          batch_size=256) -> dict:
    start_time = time()
    examples = load_data(args.valid_path, add_forward_triplet=eval_forward, add_backward_triplet=not eval_forward)

    hr_tensor, _ = predictor.predict_by_examples(examples)
    hr_tensor = hr_tensor.to(entity_tensor.device)
    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    logger.info('predict tensor done, compute metrics...')

    topk_scores, topk_indices, metrics, ranks = compute_metrics(hr_tensor=hr_tensor, entities_tensor=entity_tensor,
                                                                target=target, examples=examples,
                                                                batch_size=batch_size)
    eval_dir = 'forward' if eval_forward else 'backward'
    logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))

    pred_infos = []
    for idx, ex in enumerate(examples):
        cur_topk_scores = topk_scores[idx]
        cur_topk_indices = topk_indices[idx]
        pred_idx = cur_topk_indices[0]
        cur_score_info = {entity_dict.get_entity_by_idx(topk_idx).entity: round(topk_score, 3)
                          for topk_score, topk_idx in zip(cur_topk_scores, cur_topk_indices)}

        pred_info = PredInfo(head=ex.head, relation=ex.relation,
                             tail=ex.tail, pred_tail=entity_dict.get_entity_by_idx(pred_idx).entity,
                             pred_score=round(cur_topk_scores[0], 4),
                             topk_score_info=json.dumps(cur_score_info),
                             rank=ranks[idx],
                             correct=pred_idx == target[idx])
        pred_infos.append(pred_info)

    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    with open('{}/eval_{}_{}_{}.json'.format(prefix, split, eval_dir, basename), 'w', encoding='utf-8') as writer:
        writer.write(json.dumps([asdict(info) for info in pred_infos], ensure_ascii=False, indent=4))

    logger.info('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))
    return metrics


def _get_shard_path(shard_id=0):
    return '{}/shard_{}'.format(args.model_dir, shard_id)


def _dump_entity_embeddings(predictor: BertPredictor):
    for start in range(0, len(entity_dict), SHARD_SIZE):
        end = start + SHARD_SIZE
        shard_id = start // SHARD_SIZE
        shard_path = _get_shard_path(shard_id=shard_id)
        if os.path.exists(shard_path):
            logger.info('{} already exists'.format(shard_path))
            continue

        logger.info('shard_id={}, from {} to {}'.format(shard_id, start, end))
        shard_entity_exs = entity_dict.entity_exs[start:end]
        shard_entity_tensor = predictor.predict_by_entities(shard_entity_exs)
        torch.save(shard_entity_tensor, _get_shard_path(shard_id=shard_id))

        logger.info('done for shard_id={}'.format(shard_id))


def _load_entity_embeddings():
    assert os.path.exists(_get_shard_path())

    shard_tensors = []
    for start in range(0, len(entity_dict), SHARD_SIZE):
        shard_id = start // SHARD_SIZE
        shard_path = _get_shard_path(shard_id=shard_id)
        shard_entity_tensor = torch.load(shard_path, map_location=lambda storage, loc: storage)
        logger.info('Load {} entity embeddings from {}'.format(shard_entity_tensor.size(0), shard_path))
        shard_tensors.append(shard_entity_tensor)

    entity_tensor = torch.cat(shard_tensors, dim=0)
    logger.info('{} entity embeddings in total'.format(entity_tensor.size(0)))
    assert entity_tensor.size(0) == len(entity_dict.entity_exs)
    return entity_tensor


def predict_by_split():
    args.batch_size = max(args.batch_size, torch.cuda.device_count() * 1024)
    assert os.path.exists(args.valid_path)
    assert os.path.exists(args.train_path)
    assert os.path.exists(args.eval_model_path)

    predictor = BertPredictor()
    predictor.load(ckt_path=args.eval_model_path, use_data_parallel=True)
    _dump_entity_embeddings(predictor)

    entity_tensor = _load_entity_embeddings().cuda()
    forward_metrics = eval_single_direction(predictor,
                                            entity_tensor=entity_tensor,
                                            eval_forward=True,
                                            batch_size=32)
    backward_metrics = eval_single_direction(predictor,
                                             entity_tensor=entity_tensor,
                                             eval_forward=False,
                                             batch_size=32)
    metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
    logger.info('Averaged metrics: {}'.format(metrics))

    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    with open('{}/metrics_{}_{}.json'.format(prefix, split, basename), 'w', encoding='utf-8') as writer:
        writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
        writer.write('backward metrics: {}\n'.format(json.dumps(backward_metrics)))
        writer.write('average metrics: {}\n'.format(json.dumps(metrics)))


if __name__ == '__main__':
    predict_by_split()
