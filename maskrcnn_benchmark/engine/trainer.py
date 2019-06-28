# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist
import torch_xla
import torch_xla_py.xla_model as xm
import torch_xla_py.data_parallel as dp

from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import get_world_size, get_rank
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        # print(torch_xla._XLAC._xla_metrics_report())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        # optimizer.step()

        xm.optimizer_step(optimizer)
        print(torch_xla._XLAC._xla_metrics_report())
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 1 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        # "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    # memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

def do_train_tpu(
    model,
    cfg,
    data_loader,
    metrics_debug
):
    # Build model for distributed TPU training.
    devices = xm.get_xla_supported_devices(max_devices=cfg.NUM_CORES)
    parallel_model = dp.DataParallel(model, device_ids=devices)
    per_core_max_iter = int(len(data_loader) / cfg.NUM_CORES)

    def train_loop_fn(model, loader, device, context):
        logger = logging.getLogger("maskrcnn_benchmark.trainer")
        logger.info("starting train_loop_fn on device: {}".format(device))
        meters = MetricLogger(delimiter="  ")

        # Create optimizer and schedule lr
        optimizer = make_optimizer(cfg, model)
        scheduler = make_lr_scheduler(cfg, optimizer)

        # Bootstrap
        arguments = {}
        arguments["iteration"] = 0
        output_dir = cfg.OUTPUT_DIR
        save_to_disk = get_rank() == 0
        checkpointer = DetectronCheckpointer(
            cfg, model, optimizer, scheduler, output_dir, save_to_disk
        )
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
        arguments.update(extra_checkpoint_data)
        start_training_time = time.time()
        end = time.time()
        tracker = xm.RateTracker()

        for iteration, (images, targets, _) in loader:
            data_time = time.time() - end
            iteration += 1
            arguments["iteration"] = iteration

            scheduler.step()
            images = images.to(device)
            targets = [target.to(device) for target in targets]

            loss_dict = model(images, targets)
            # l = l_mask + l_bbox + l_class
            losses = sum(loss for loss in loss_dict.values())

            # No need for TPUs reduce losses over devices as we loop per device.
            # loss_dict_reduced = reduce_loss_dict(loss_dict)
            # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses, **loss_dict)

            optimizer.zero_grad()
            losses.backward()
            xm.optimizer_step(optimizer)

            tracker.add(cfg.SOLVER.IMS_PER_BATCH)
            batch_time = time.time() - end
            end = time.time()
            if metrics_debug:
                print(torch_xla._XLAC._xla_metrics_report())

            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg*(per_core_max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 1 == 0 or iteration == per_core_max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "[{device}]({iter}/{per_core_max_iter})",
                            "eta: {eta}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "time_elapsed_sec: {time_elapsed:.2f}",
                            "rate: {rate:.2} img/sec",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        time_elapsed=time.time()-start_training_time,
                        device=device,
                        rate=tracker.rate(),
                        per_core_max_iter=per_core_max_iter,
                    )
                )
            if iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if iteration == per_core_max_iter:
                checkpointer.save("model_final", **arguments)

    result = parallel_model(train_loop_fn, data_loader)

