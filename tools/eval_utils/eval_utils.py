import pickle
import time
import sys
import numpy as np
import torch
import tqdm

from btcdet.models import load_data_to_gpu
from btcdet.utils import common_utils
from ptflops.flops_counter import add_flops_counting_methods


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def get_match_stats(ret_dict, metric):
    pos_num, neg_num, total, precision, recall, f1, pos_all_num = ret_dict["pos_num"], ret_dict["neg_num"], ret_dict["total"], ret_dict["precision"], ret_dict["recall"], ret_dict["f1"], ret_dict["pos_all_num"]
    total_factor = total / 1000.0
    metric["scene_total_factor"] += total_factor
    metric["scene_num"] += 1
    metric["precision"] += precision
    metric["recall"] += recall
    metric["f1"] += f1
    metric["precision_factored"] += precision * total_factor
    metric["recall_factored"] += recall * total_factor
    metric["f1_factored"] += f1 * total_factor
    metric["total_pos_all_portion"] += pos_num / max(1.0, pos_all_num)

    box_num_sum, occ_box_num = ret_dict["box_num_sum"], ret_dict["occ_box_num"]
    metric["total_num_box"] += box_num_sum
    for cur_thresh in range(1,10):
        metric['total_occ_num_box_%.1f' % (cur_thresh*0.1)] += occ_box_num[cur_thresh-1]

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, pc_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
        'scene_num': 0,
        'scene_total_factor': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'precision_factored': 0,
        'recall_factored': 0,
        'f1_factored': 0,
        'total_num_box': 0,
        'total_pos_all_portion': 0,
    }

    if hasattr(cfg.MODEL, "OCC") and hasattr(cfg.MODEL.OCC, "OCC_POST_PROCESSING"):
        for cur_thresh in range(1,10):
            metric['total_occ_num_box_%.1f' % (cur_thresh*0.1)] = 0

    if hasattr(cfg.MODEL, "POST_PROCESSING"):
        for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
            metric['recall_roi_%s' % str(cur_thresh)] = 0
            metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    tb_dict_valid={}
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        #
        # flops_model = add_flops_counting_methods(model)
        # flops_model.__batch_counter__ = 1
        # flops_model.eval()
        # flops_model.start_flops_count(ost=sys.stdout, verbose=True,
        #                               ignore_list=[])
        # macs, params = flops_model.compute_average_flops_cost()
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        with torch.no_grad():
            model.global_step = torch.tensor(i, dtype=torch.int64)
            pred_dicts, ret_dict, tb_dict, pc_dict = model(batch_dict)
        disp_dict = {}
        if hasattr(cfg.MODEL, "OCC") and hasattr(cfg.MODEL.OCC, "OCC_POST_PROCESSING"):
            get_match_stats(ret_dict, metric)
        if hasattr(cfg.MODEL, "POST_PROCESSING"):
            statistics_info(cfg, ret_dict, metric, disp_dict)
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=final_output_dir if save_to_file else None
            )
            det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

            if bool(pc_dict):
                np.save(str(pc_dir)+'/pc_{}_{}'.format(epoch_id, i), pc_dict)
            if bool(tb_dict):
                tb_dict_valid.update(tb_dict)
        # print("torch.get_num_threads", torch.get_num_threads())

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    if hasattr(cfg.MODEL, "POST_PROCESSING"):
        for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
            cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
            logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
            ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
            ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall
    if hasattr(cfg.MODEL, "OCC") and hasattr(cfg.MODEL.OCC, "OCC_POST_PROCESSING"):
        logger.info(" *************************** ")
        logger.info(' '.join([
            f"precision: {metric['precision'] / metric['scene_num']:.3f}, recall: {metric['recall'] / metric['scene_num']:.3f},", f"f1: {metric['f1'] / metric['scene_num']:.3f}, precision_factored: {metric['precision_factored'] / metric['scene_total_factor']:.3f}", f"recall_factored: {metric['recall_factored'] / metric['scene_total_factor']:.3f}, f1_factored: {metric['f1_factored'] / metric['scene_total_factor']:.3f}"
        ]))
        logger.info(' '.join([
            f"occ thresh {i * 0.1:.1f}: {metric['total_occ_num_box_%.1f' % (i*0.1)] / metric['total_num_box']:.3f},  " for i in range(1,10)]))
        logger.info(' '.join([f" total_pos_all_portion {metric['total_pos_all_portion'] / metric['scene_num']:.3f}"]))

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    if hasattr(cfg.MODEL, "POST_PROCESSING"):
        result_str, result_dict, pr_rc_details = dataset.evaluation(
            det_annos, class_names,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=final_output_dir,
            coverage_rates= cfg.MODEL.POST_PROCESSING.get("CVRG_RATES", None)
        )

        logger.info(result_str)
        ret_dict.update(result_dict)

        logger.info('Result is save to %s' % result_dir)

        # logger.info("\n -------- pr rc breakdown -------------- \n")
        # logger.info(pr_rc_details)
        # logger.info("\n")
        #
        # with open(result_dir / 'pc_rc.pkl', 'wb') as f:
        #     pickle.dump(pr_rc_details, f)
        # logger.info('pr rc break down is save to %s' % result_dir)

        logger.info('****************Evaluation done.*****************')
    ret_dict.update(tb_dict_valid)
    return ret_dict


if __name__ == '__main__':
    pass
