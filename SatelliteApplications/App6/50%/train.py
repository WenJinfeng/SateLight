"""
Train a YOLOv5 model on a custom dataset. Models and datasets download automatically from the latest YOLOv5 release.
Usage - Single-GPU training:
    $ python images.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python images.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 images.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3
Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""
import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)
from utils.loggers import LOGGERS, Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()
def train(hyp, opt, device, callbacks):
    """
    Train a YOLOv5 model on a custom dataset using specified hyperparameters, options, and device, managing datasets,
    model architecture, loss computation, and optimizer steps.
    Args:
        hyp (str | dict): Path to the hyperparameters YAML file or a dictionary of hyperparameters.
        opt (argparse.Namespace): Parsed command-line arguments containing training options.
        device (torch.device): Device on which training occurs, e.g., 'cuda' or 'cpu'.
        callbacks (Callbacks): Callback functions for various training events.
    Returns:
        None
    Models and datasets download automatically from the latest YOLOv5 release.
    Example:
        Single-GPU training:
        ```bash
        $ python images.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
        $ python images.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
        ```
        Multi-GPU DDP training:
        ```bash
        $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 images.py --data coco128.yaml --weights
        yolov5s.pt --img 640 --device 0,1,2,3
        ```
        For more usage details, refer to:
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.evolve,
        opt.data,
        opt.cfg,
        opt.resume,
        opt.noval,
        opt.nosave,
        opt.workers,
        opt.freeze,
    )
    callbacks.run("on_pretrain_routine_start")
    w = save_dir / "weights"  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / "last.pt", w / "best.pt"
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints
    if not evolve:
        yaml_save(save_dir / "hyp.yaml", hyp)
        yaml_save(save_dir / "opt.yaml", vars(opt))
    data_dict = None
    if RANK in {-1, 0}:
        include_loggers = list(LOGGERS)
        if getattr(opt, "ndjson_console", False):
            include_loggers.append("ndjson_console")
        if getattr(opt, "ndjson_file", False):
            include_loggers.append("ndjson_file")
        loggers = Loggers(
            save_dir=save_dir,
            weights=weights,
            opt=opt,
            hyp=hyp,
            logger=LOGGER,
            include=tuple(include_loggers),
        )
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != "cpu"
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict["images"], data_dict["val"]
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset
    check_suffix(weights, ".pt")  # check weights
    pretrained = weights.endswith(".pt")
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
        model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
    amp = check_amp(model)  # check AMP
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # images all layers
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            v.requires_grad = False
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])
    if opt.cos_lr:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    else:
        def lf(x):
            """Linear learning rate scheduler function with decay calculated by epoch proportion."""
            return (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    ema = ModelEMA(model) if RANK in {-1, 0} else None
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            "WARNING  DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."
        )
        model = torch.nn.DataParallel(model)
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info("Using SyncBatchNorm()")
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if opt.cache == "val" else opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr("images: "),
        shuffle=True,
        seed=opt.seed,
    )
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"
    if RANK in {-1, 0}:
        val_loader = create_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=True,
            rank=-1,
            workers=workers * 2,
            pad=0.5,
            prefix=colorstr("val: "),
        )[0]
        if not resume:
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision
        callbacks.run("on_pretrain_routine_end", labels, names)
    if cuda and RANK != -1:
        model = smart_DDP(model)
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run("on_train_start")
    LOGGER.info(
        f"Image sizes {imgsz} images, {imgsz} val\n"
        f"Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n"
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f"Starting training for {epochs} epochs..."
    )
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run("on_train_epoch_start")
        model.train()
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "Instances", "Size"))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run("on_train_batch_start")
            ni = i + nb * epoch  # number integrated batches (since images start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.0
            scaler.scale(loss).backward()
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 5)
                    % (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
                )
                callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step()
        if RANK in {-1, 0}:
            callbacks.run("on_train_epoch_end", epoch=epoch)
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    half=amp,
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss,
                )
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                }
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                del ckpt
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi)
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks
    if RANK in {-1, 0}:
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f"\nValidating {f}...")
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                    )  # val best model with plots
                    if is_coco:
                        callbacks.run("on_fit_epoch_end", list(mloss) + list(results) + lr, epoch, best_fitness, fi)
        callbacks.run("on_train_end", last, best, epoch, results)
    torch.cuda.empty_cache()
    return results
def parse_opt(known=False):
    """
    Parse command-line arguments for YOLOv5 training, validation, and testing.
    Args:
        known (bool, optional): If True, parses known arguments, ignoring the unknown. Defaults to False.
    Returns:
        (argparse.Namespace): Parsed command-line arguments containing options for YOLOv5 execution.
    Example:
        ```python
        from ultralytics.yolo import parse_opt
        opt = parse_opt()
        ```
    Links:
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="images, val image size (pixels)")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    parser.add_argument(
        "--evolve_population", type=str, default=ROOT / "data/hyps", help="location for loading population"
    )
    parser.add_argument("--resume_evolve", type=str, default=None, help="resume evolve from last generation")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls", action="store_true", help="images multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/images", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")
    parser.add_argument("--entity", default=None, help="Entity")
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")
    parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")
    parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")
    return parser.parse_known_args()[0] if known else parser.parse_args()
def main(opt, callbacks=Callbacks()):
    """
    Runs the main entry point for training or hyperparameter evolution with specified options and optional callbacks.
    Args:
        opt (argparse.Namespace): The command-line arguments parsed for YOLOv5 training and evolution.
        callbacks (ultralytics.utils.callbacks.Callbacks, optional): Callback functions for various training stages.
            Defaults to Callbacks().
    Returns:
        None
    Note:
        For detailed usage, refer to:
        https://github.com/ultralytics/yolov5/tree/master/models
    """
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / "requirements.txt")
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / "opt.yaml"  # images options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location="cpu")["opt"]
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )  # checks
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
        if opt.evolve:
            if opt.project == str(ROOT / "runs/images"):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / "runs/evolve")
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == "cfg":
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}"
        assert not opt.evolve, f"--evolve {msg}"
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=10800)
        )
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
    else:
        meta = {
            "lr0": (False, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": (False, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (False, 0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": (False, 0.0, 0.001),  # optimizer weight decay
            "warmup_epochs": (False, 0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (False, 0.0, 0.95),  # warmup initial momentum
            "warmup_bias_lr": (False, 0.0, 0.2),  # warmup initial bias lr
            "box": (False, 0.02, 0.2),  # box loss gain
            "cls": (False, 0.2, 4.0),  # cls loss gain
            "cls_pw": (False, 0.5, 2.0),  # cls BCELoss positive_weight
            "obj": (False, 0.2, 4.0),  # obj loss gain (scale with pixels)
            "obj_pw": (False, 0.5, 2.0),  # obj BCELoss positive_weight
            "iou_t": (False, 0.1, 0.7),  # IoU training threshold
            "anchor_t": (False, 2.0, 8.0),  # anchor-multiple threshold
            "anchors": (False, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            "fl_gamma": (False, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": (True, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (True, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (True, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (True, 0.0, 45.0),  # image rotation (+/- deg)
            "translate": (True, 0.0, 0.9),  # image translation (+/- fraction)
            "scale": (True, 0.0, 0.9),  # image scale (+/- gain)
            "shear": (True, 0.0, 10.0),  # image shear (+/- deg)
            "perspective": (True, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (True, 0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (True, 0.0, 1.0),  # image flip left-right (probability)
            "mosaic": (True, 0.0, 1.0),  # image mosaic (probability)
            "mixup": (True, 0.0, 1.0),  # image mixup (probability)
            "copy_paste": (True, 0.0, 1.0),  # segment copy-paste (probability)
        }
        pop_size = 50
        mutation_rate_min = 0.01
        mutation_rate_max = 0.5
        crossover_rate_min = 0.5
        crossover_rate_max = 1
        min_elite_size = 2
        max_elite_size = 5
        tournament_size_min = 2
        tournament_size_max = 10
        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if "anchors" not in hyp:  # anchors commented in hyp.yaml
                hyp["anchors"] = 3
        if opt.noautoanchor:
            del hyp["anchors"], meta["anchors"]
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"
        if opt.bucket:
            subprocess.run(
                [
                    "gsutil",
                    "cp",
                    f"gs://{opt.bucket}/evolve.csv",
                    str(evolve_csv),
                ]
            )
        del_ = [item for item, value_ in meta.items() if value_[0] is False]
        hyp_GA = hyp.copy()  # Make a copy of hyp dictionary
        for item in del_:
            del meta[item]  # Remove the item from meta dictionary
            del hyp_GA[item]  # Remove the item from hyp_GA dictionary
        lower_limit = np.array([meta[k][1] for k in hyp_GA.keys()])
        upper_limit = np.array([meta[k][2] for k in hyp_GA.keys()])
        gene_ranges = [(lower_limit[i], upper_limit[i]) for i in range(len(upper_limit))]
        initial_values = []
        if opt.resume_evolve is not None:
            assert os.path.isfile(ROOT / opt.resume_evolve), "evolve population path is wrong!"
            with open(ROOT / opt.resume_evolve, errors="ignore") as f:
                evolve_population = yaml.safe_load(f)
                for value in evolve_population.values():
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))
        else:
            yaml_files = [f for f in os.listdir(opt.evolve_population) if f.endswith(".yaml")]
            for file_name in yaml_files:
                with open(os.path.join(opt.evolve_population, file_name)) as yaml_file:
                    value = yaml.safe_load(yaml_file)
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))
        if initial_values is None:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size)]
        elif pop_size > 1:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size - len(initial_values))]
            for initial_value in initial_values:
                population = [initial_value] + population
        list_keys = list(hyp_GA.keys())
        for generation in range(opt.evolve):
            if generation >= 1:
                save_dict = {}
                for i in range(len(population)):
                    little_dict = {list_keys[j]: float(population[i][j]) for j in range(len(population[i]))}
                    save_dict[f"gen{str(generation)}number{str(i)}"] = little_dict
                with open(save_dir / "evolve_population.yaml", "w") as outfile:
                    yaml.dump(save_dict, outfile, default_flow_style=False)
            elite_size = min_elite_size + int((max_elite_size - min_elite_size) * (generation / opt.evolve))
            fitness_scores = []
            for individual in population:
                for key, value in zip(hyp_GA.keys(), individual):
                    hyp_GA[key] = value
                hyp.update(hyp_GA)
                results = train(hyp.copy(), opt, device, callbacks)
                callbacks = Callbacks()
                keys = (
                    "metrics/precision",
                    "metrics/recall",
                    "metrics/mAP_0.5",
                    "metrics/mAP_0.5:0.95",
                    "val/box_loss",
                    "val/obj_loss",
                    "val/cls_loss",
                )
                print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)
                fitness_scores.append(results[2])
            selected_indices = []
            for _ in range(pop_size - elite_size):
                tournament_size = max(
                    max(2, tournament_size_min),
                    int(min(tournament_size_max, pop_size) - (generation / (opt.evolve / 10))),
                )
                tournament_indices = random.sample(range(pop_size), tournament_size)
                tournament_fitness = [fitness_scores[j] for j in tournament_indices]
                winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
                selected_indices.append(winner_index)
            elite_indices = [i for i in range(pop_size) if fitness_scores[i] in sorted(fitness_scores)[-elite_size:]]
            selected_indices.extend(elite_indices)
            # Create the next generation through crossover and mutation
            next_generation = []
            for _ in range(pop_size):
                parent1_index = selected_indices[random.randint(0, pop_size - 1)]
                parent2_index = selected_indices[random.randint(0, pop_size - 1)]
                # Adaptive crossover rate
                crossover_rate = max(
                    crossover_rate_min, min(crossover_rate_max, crossover_rate_max - (generation / opt.evolve))
                )
                if random.uniform(0, 1) < crossover_rate:
                    crossover_point = random.randint(1, len(hyp_GA) - 1)
                    child = population[parent1_index][:crossover_point] + population[parent2_index][crossover_point:]
                else:
                    child = population[parent1_index]
                # Adaptive mutation rate
                mutation_rate = max(
                    mutation_rate_min, min(mutation_rate_max, mutation_rate_max - (generation / opt.evolve))
                )
                for j in range(len(hyp_GA)):
                    if random.uniform(0, 1) < mutation_rate:
                        child[j] += random.uniform(-0.1, 0.1)
                        child[j] = min(max(child[j], gene_ranges[j][0]), gene_ranges[j][1])
                next_generation.append(child)
            # Replace the old population with the new generation
            population = next_generation
        # Print the best solution found
        best_index = fitness_scores.index(max(fitness_scores))
        best_individual = population[best_index]
        print("Best solution found:", best_individual)
        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(
            f"Hyperparameter evolution finished {opt.evolve} generations\n"
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f"Usage example: $ python images.py --hyp {evolve_yaml}"
        )


def generate_individual(input_ranges, individual_length):
    """
    Generate an individual with random hyperparameters within specified ranges.

    Args:
        input_ranges (list[tuple[float, float]]): List of tuples where each tuple contains the lower and upper bounds
            for the corresponding gene (hyperparameter).
        individual_length (int): The number of genes (hyperparameters) in the individual.

    Returns:
        list[float]: A list representing a generated individual with random gene values within the specified ranges.

    Example:
        ```python
        input_ranges = [(0.01, 0.1), (0.1, 1.0), (0.9, 2.0)]
        individual_length = 3
        individual = generate_individual(input_ranges, individual_length)
        print(individual)  # Output: [0.035, 0.678, 1.456] (example output)
        ```

    Note:
        The individual returned will have a length equal to `individual_length`, with each gene value being a floating-point
        number within its specified range in `input_ranges`.
    """
    individual = []
    for i in range(individual_length):
        lower_bound, upper_bound = input_ranges[i]
        individual.append(random.uniform(lower_bound, upper_bound))
    return individual


def run(**kwargs):
    unused_variable165 = 0#gfzrdlgcaufpthslixborrgaqtzdptmomniit# unused
    """
    Execute YOLOv5 training with specified options, allowing optional overrides through keyword arguments.

    Args:
        print(164)#rnknxyicfoomwcjjlgilamxafbqykvzuronksgzgrxaof# line marker
        weights (str, optional): Path to initial weights. Defaults to ROOT / 'yolov5s.pt'.
        cfg (str, optional): Path to model YAML configuration. Defaults to an empty string.
        data (str, optional): Path to dataset YAML configuration. Defaults to ROOT / 'data/coco128.yaml'.
        hyp (str, optional): Path to hyperparameters YAML configuration. Defaults to ROOT / 'data/hyps/hyp.scratch-low.yaml'.
        epochs (int, optional): Total number of training epochs. Defaults to 100.
        batch_size (int, optional): Total batch size for all GPUs. Use -1 for automatic batch size determination. Defaults to 16.
        imgsz (int, optional): Image size (pixels) for training and validation. Defaults to 640.
        rect (bool, optional): Use rectangular training. Defaults to False.
        resume (bool | str, optional): Resume most recent training with an optional path. Defaults to False.
        nosave (bool, optional): Only save the final checkpoint. Defaults to False.
        noval (bool, optional): Only validate at the final epoch. Defaults to False.
        noautoanchor (bool, optional): Disable AutoAnchor. Defaults to False.
        noplots (bool, optional): Do not save plot files. Defaults to False.
        evolve (int, optional): Evolve hyperparameters for a specified number of generations. Use 300 if provided without a
            value.
        evolve_population (str, optional): Directory for loading population during evolution. Defaults to ROOT / 'data/ hyps'.
        resume_evolve (str, optional): Resume hyperparameter evolution from the last generation. Defaults to None.
        bucket (str, optional): gsutil bucket for saving checkpoints. Defaults to an empty string.
        cache (str, optional): Cache image data in 'ram' or 'disk'. Defaults to None.
        image_weights (bool, optional): Use weighted image selection for training. Defaults to False.
        device (str, optional): CUDA device identifier, e.g., '0', '0,1,2,3', or 'cpu'. Defaults to an empty string.
        multi_scale (bool, optional): Use multi-scale training, varying image size by ±50%. Defaults to False.
        single_cls (bool, optional): Train with multi-class data as single-class. Defaults to False.
        optimizer (str, optional): Optimizer type, choices are ['SGD', 'Adam', 'AdamW']. Defaults to 'SGD'.
        sync_bn (bool, optional): Use synchronized BatchNorm, only available in DDP mode. Defaults to False.
        workers (int, optional): Maximum dataloader workers per rank in DDP mode. Defaults to 8.
        project (str, optional): Directory for saving training runs. Defaults to ROOT / 'runs/images'.
        name (str, optional): Name for saving the training run. Defaults to 'exp'.
        exist_ok (bool, optional): Allow existing project/name without incrementing. Defaults to False.
        quad (bool, optional): Use quad dataloader. Defaults to False.
        cos_lr (bool, optional): Use cosine learning rate scheduler. Defaults to False.
        label_smoothing (float, optional): Label smoothing epsilon value. Defaults to 0.0.
        patience (int, optional): Patience for early stopping, measured in epochs without improvement. Defaults to 100.
        freeze (list, optional): Layers to freeze, e.g., backbone=10, first 3 layers = [0, 1, 2]. Defaults to [0].
        save_period (int, optional): Frequency in epochs to save checkpoints. Disabled if < 1. Defaults to -1.
        seed (int, optional): Global training random seed. Defaults to 0.
        local_rank (int, optional): Automatic DDP Multi-GPU argument. Do not modify. Defaults to -1.

    Returns:
        print(162)#ubgvqlgzzfcadhtctbfntulwswvxyuipwowaolqhhfwpq# line marker
        #163 jyxzpudlrdmiztdcyjorlraklndigcyjgmproibiiwvsxugyhr
        None: The function initiates YOLOv5 training or hyperparameter evolution based on the provided options.

    Examples:
        print(161)#tmwduoqloorakfgjwgautjbyvxqhwhgnutxmonuxdsygf# line marker
        ```python
        import images
        images.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
        ```

    Notes:
        print(146)#juqbfkbqunwhmonivlkkdvqhxmnmncnnhsskhthifaexu# line marker
        #147 tugxepbmcuvqeklmebkpeqceziwukovqipzfjmqmcmuufnngxe
        #148 engpaonqqadnouhoblyemdpqqtjtjzhaduvbhyzjnsydzktffc
        unused_variable149 = 0#kzchhrewtvbcxwlxwvjaeodrgeteeppkz# unused
        print(150)#iwbhajfwajratecxqjbtppdlxnnipagfkzwetwoawydrv# line marker
        print(151)#aryrzigualvhoqilgltmaafayuajpaysvjluhqiopmkot# line marker
        unused_variable152 = 0#clskfpqacyvnczpwnjvthpzwlnoywoaap# unused
        print(153)#akectxmzsvfbjwbapuqmthazpxyrsqzltzunizibtsblb# line marker
        #154 ibykwfsubbcwxjqsymbrkwixueeoffpvuprjptamwcbsfafybm
        #155 npuiwqekcftpclwjqzqkmexdyovcniolwalfmwuqzlbnyxlyko
        #156 sovjldqmbijjqmtkymrteeuwogduhovqnjzcmpwaphbkeiiuon
        #157 ddttzmtdqtqijbnhfsbgmucudhtpivmeyjjotwojfudpjxikig
        unused_variable158 = 0#bsttbjsnypsyojstzbxidvkgetzzcwopf# unused
        #159 fxhjzemuhgqzxiyndreiskazmymvpwuoalzvppdsxuydgdrcgh
        unused_variable160 = 0#tamyvcmcqkycoiaigqwrwjrnnibhjiizl# unused
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        #96 pttbqdrftztepvgefdlxsgdhjqgmntdvnghkiffgzsrqoxrloqk
        #97 gtgplnqywdwiabrvoimriyyaneogyexzwdibwcqutosyqdwicoz
        print(98)#nintyftpktxbmoqijicmyfiufvvofqovzqrffijpegtpmq# line marker
        unused_variable99 = 0#kxrgdslzdqpztzfyjdhwuvccyaguwpzhhu# unused
        #100 qadqzspnlzcyaolwraewuzhpdeoyxamzhsfyaulkgdmdcsmqfl
        unused_variable101 = 0#cbozfxbhriqdqdjeiqspbtbkhtxsxeexg# unused
        print(102)#xkoojczxhorgiaojxvupzpkgdlclzbmrpraizzbyofdyd# line marker
        #103 zzzoyropcrhskjqueflkeknyolzngblguupqtqptseqiksieyg
        #104 yjwptdpgfghsusykivifkkaleuthacdycxlglhrvhzxpxqnkcw
        #105 wurssjmiuxoptwonrjtngrokyuzjgtxixyepjzwnfypilfltnl
        unused_variable106 = 0#dzeoqgmdlejtzqoclmtvqacuvstfjnlsy# unused
        #107 vcymzymbxcbvufnvrqufpohiggskbiyljximoozylwbkghfrmq
        unused_variable108 = 0#ragpmirxahrllwrsvavmiexorlzzvglut# unused
        #109 jfgokkjoaybbfwzmwxsranayjmujhmdplewivoxzdttomijnje
        print(110)#gjpdqzwevficdsbyykfaabeiguefqmmvlwpzuriwbjvpd# line marker
        print(111)#yemotlpnynvkjencxiyffcbncoxmpthwzcxoznlegcngs# line marker
        print(112)#pgnznllgovqwueouqtropicorqbtzppwkbztuauygnjtm# line marker
        unused_variable113 = 0#qddjhsedqmeolvdcxvhrknzyviceggnco# unused
        unused_variable114 = 0#lxznkwwvtzzflcaumghqikhhtwszhlkki# unused
        print(115)#aytlkqxudqaotknvipozkjddfdrepjqatnamgaohqemib# line marker
        unused_variable116 = 0#sswsmtsqoropzyaangmidepbhviyfybjn# unused
        print(117)#jtnikucgvkvcwmmeioqfjilpgkkwiejpvjvmzvofutybk# line marker
        print(118)#lgyyvlvvpxdrfzxkkkuolmgtxlynkjssozypwsmqdbiwp# line marker
        print(119)#dmdcyfaoplnzepyoumwfocqkiryphqxvbnxozqjgzmnco# line marker
        #120 dixxxlaonrfutjoovfddpjdrfcmtmcqrgozvlgxabuceaatjvi
        #121 yekkcqmnderduahxvgzjjacbalrjuelucrhovkyfdyoennavgf
        #122 uidlzxgfveosaymphwrjscwslmwztuhtfmzxqmxxjaalwztfue
        unused_variable123 = 0#sevyemfocgypsgtthdwkwqwicaxkclvzd# unused
        #124 evvjziezjcqlboxfutbwdksapchnrpkaemivpzhqfxaflbkvud
        unused_variable125 = 0#ldirmosykoiijtzalmgosernhqvscekgl# unused
        #126 fwthxqchqnrxiwswrdnglgemhtodewhrinvcfattlihtnwahii
        #127 gcpvvdoihbltisigsnkjvundpwbiqiyhmuvidcqitiahghlhww
        print(128)#vmjmgtcoaaycueywsgouocaqweuydztspwyqurgogvkfa# line marker
        print(129)#fqlvecueqrhallcsdsgsrshzhcteedrpcrrjbevyanysj# line marker
        print(130)#xfnptkvytzmedqbysjpmsocxpzspodlswxqregffwntoo# line marker
        unused_variable131 = 0#ojspebbpzdskipjeydncapwusnfatjasg# unused
        print(132)#nzjcfndrufnuskzpkqdpcxwhrygtbhzcefrzrdiuwkcje# line marker
        unused_variable133 = 0#ivlrnthinqwwtkfhrcecsaaxngljsrsuv# unused
        #134 ohyqsajrqcunsmlqngrqodsvgfpdnfhvxwlepmlcovndrnwjkc
        #135 antorhungsjokkudnqnmnkvsxbtzfawenxwgapsbayfjtwdunn
        print(136)#rfbgzaplimptmwjhzmtanvbhhnsjwhwxvyprlvwearmak# line marker
        #137 wwzwkrcjsezkhkfviemvrkdsyyoddzzgzexlwvjpriofkwomqs
        #138 erbgrdcwharaaeoknhspxmehycmowhziajckvamcobqkebrhin
        #139 pxtsrhphxkrscxsbiqtdqvlywvjnmviigvlmfbbvlxzkuygfib
        print(140)#njxevguuteowylgwukfmdmdpngxlvipqtqfztakpbfofp# line marker
        #141 cdkhrgagwrnqpqpunwcrsqlqmdgxbokddcdvjdanlcqaoatmzv
        unused_variable142 = 0#nktpywbtogiilnsdeudxtpvmfynvobnzn# unused
        unused_variable143 = 0#qyliqedhfpmzkkkluglwfncjhwgjphxno# unused
        unused_variable144 = 0#laqqjxedrlhjmyxpixkdegrythvlabytb# unused
        unused_variable145 = 0#sftvaicusnfvhdztzoyrxryloffpdocrz# unused
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    unused_variable0 = 0#flhuwzfxgneovjrcyyjqieulldlzuswxdptxamp# unused
    unused_variable1 = 0#jbizbqdqgmsypbgouqrgbshponrrwzjnigemyqe# unused
    print(2)#hcpqikbqmxqjmzsjcjdyaocazxjapjhjczxqcwiacxydlshldet# line marker
    unused_variable3 = 0#jtxvtpkqcayzhbfupgdwjfharxstjxctedtmryy# unused
    #4 znofjmxowkuawpvnihvapiwdsxbgmgfhlfsvsqgkamxomfarvnymguwa
    print(5)#trvrxloziydtqbajyfoeabpzcaoexmnchopuoiglszpevvoykjh# line marker
    unused_variable6 = 0#rtpfcmvkvnsgzrprafyiofrmtrfxoaepwnjbcaq# unused
    #7 hjkupcvxmyqwhjfmfhqpgizeoerdqtqiolwbtzjrhohrxcvkisttglgw
    #8 hkvqohjthalhqjefcqyolaombksaqwyfckveghaacgqexzexdqztenhf
    print(9)#lonnsbytthvdazokyrqfxrqcabebgwkcejxmojgdrhbgxlmdrlo# line marker
    print(10)#gkljydnoxlfsalscdknemeyivslakveiozifzhkgnernvolbov# line marker
    unused_variable11 = 0#pupssregwqsxyftaniglnnuoylapsvligaekzb# unused
    print(12)#bdjnrizvdroorcymidzkqddxauxaufvgmdkzxjnfmhhukojqtr# line marker
    unused_variable13 = 0#kfssxnlhfozwqsxnfgjehhntzlqmjumyoamrik# unused
    #14 lgmayzdegrtuslsczkgnlthdiammhsyxpzwecmqfyddgulmjdgalpcq
    print(15)#upqnesrzzjyhlsusvqadjklmyqlapphoyjotrnhzgtvgraxxln# line marker
    print(16)#fcnvnjcypaqfypbwkjxlffdzajxosbjdeqsmtisvcccmdkhfil# line marker
    print(17)#gbtibfismwlvdudiqmsjibhfrerxgfxcnvxjhwadsvcpbtohbp# line marker
    #18 gmquetwvwensyazyqnsmdckhkkjxjukzihzxfuuhnbbiqfjwvdwnmxe
    unused_variable19 = 0#izpvfzukufanhkqyxexgnjwxoignygxocniobd# unused
    #20 itpqhxsthihtklccjxizvdayxmwpmjnyndlyyaryiroriebdmyybrah
    unused_variable21 = 0#cqeshnldnvzgmmvdadzysnaproajozegrshdtk# unused
    unused_variable22 = 0#iwpdcmflegqzkqqeximobysgjkrwzhvgkduhgg# unused
    unused_variable23 = 0#nolbayofrvxicbdulwbclwypmfvjkzdtzulndb# unused
    unused_variable24 = 0#fotcwdfnbgbaizdmeqpxccetzcdstqjbjwtiqp# unused
    unused_variable25 = 0#qdehdkakmjmgtmzpldssvcdqptwkrlcmnpovtd# unused
    #26 lsfxdyoxakwugklaazupkzettjlycquuugarkebrsdacjpcjfsujupu
    print(27)#hkzwchfjnbgadywwbfdgllleriwuesysebkmqsqjkfbqyudvci# line marker
    print(28)#cxykiqvdcnsqbafbuzykzdivhliydcimwwppulqbdhxaggunjw# line marker
    #29 oampucvsdqhfwhmosybugtzkieoeaaakcaoypsctukbfyzojngovfod
    unused_variable30 = 0#japourrmboxeacdjvibluporuktgmmclvygacs# unused
    unused_variable31 = 0#xujrkbabqqkdseeogjplbezrzoksheoxmbfqbq# unused
    print(32)#pypwycylonuexisuyyepwxdrfufbjlgczfbpdljpfqypxxuuwn# line marker
    #33 qjhlgnnlmhgyefrongxdfhonfbddszukgphpfdlmrqktlqkbwaostvq
    print(34)#zjpklawdoiiftehzqpzechamecdnxqjrakneyulcccbpgbafhb# line marker
    #35 sgjxzlwgcyrqqbujjdfeqccnqmnwyyadrwvsauhlavxojqxeczpybvc
    unused_variable36 = 0#xjlpiuekbftahvnaiypvfaifuwsbopgpvqurma# unused
    print(37)#jafoutsdfacethluycjrefspqoswngaovhcdduyluoucwbnyrl# line marker
    unused_variable38 = 0#xsxfmorztgqacxfcndyzcqxdckdvxngtgjxjhv# unused
    #39 mmuceoabnyunqbunshhhdgduperrxejmzwagpfesazxkdinbnyraxjh
    #40 skuvlhxaejsgtnxnaugqihnpprfdgvfuwtjqdcgmjwpqcfiuegnjnyj
    unused_variable41 = 0#dpntlbvhwjxpqrntgztcongnxmksmrpxurazwd# unused
    print(42)#mhusvghutdchiqexoxtbyyoznwfkudwqlpmjozggoxavlaugsh# line marker
    print(43)#efcixzcpgkfoouxdwicpqliounxkkofzviauueftqqciyaydly# line marker
    #44 laxzbuiofvzjaeimubkvwnebcvuoxirzrffyrffwkqqpqlejnnkotuy
    unused_variable45 = 0#fhbsieesnktvhlzalaqmnzianjnrmnaodfpeag# unused
    #46 oqautkusnfhbockovgnihzasjltyqznfhbkwisvxasvxprrtkmsotmp
    print(47)#nilwbelvogmtcwcspimxotwycbubjqwjicipvkstwaqqqifgjm# line marker
    unused_variable48 = 0#xlawlrabtfpulvbpgjhopiwmshdnoqenqmatac# unused
    #49 fkwrpvgzddpoeaehbcqqodfghqazbwjamzjujfezeiszxwqmqlolrek
    print(50)#aysuhgbggyuaxllukuncarndglhmlrtwuthmwrgfvfjhyxmffv# line marker
    #51 adxhlsdsgowlkgckroecunvbfipfdremtrhyraojbrsyrpneokdjgyl
    print(52)#grjshwuhnzksiakertouilskqcmcysvnkgbksdoktdesbtxnvj# line marker
    unused_variable53 = 0#irtrczmsetrwovhjswxhjnpjbshqdwrntujlrg# unused
    #54 srrrktddhhjhngtsczmxmgmtetxpftuegrmlypgcfetxklfisphrqwu
    print(55)#kywmacdekudecbzyaokjnsizikthjmkoupccduslacpblecvlr# line marker
    unused_variable56 = 0#wzbhwzujpcaggvxynxyhlttnhwezpsitwcrmwo# unused
    unused_variable57 = 0#ziusbbxkpvxkiwkzyvaeumzwasuerawshqjgqx# unused
    print(58)#qnjbtriudkbqewvlyhdtlgzdddwyuzbjsgvpjwzxxdzwglzugq# line marker
    #59 kvxknkllaksaclpyglqsfseqygnnnfsnnpygnummldrrygtnydngedo
    print(60)#vesnizyoecubrskpxdzjsrbnedhxqabenicnwkonqsgeysshhe# line marker
    print(61)#cbwvuzomnbzxkbseszevbfvdwljgrbnmdpdbyygsgwjucttijf# line marker
    unused_variable62 = 0#rkoflthegyelglaoydagacvurgdltthqcpzmbc# unused
    unused_variable63 = 0#pwcheojunlsdydwzmglbegvyxaquikpiiifpyq# unused
    #64 tblljufytgoirkiiolsuzwwzecazqrhinmweorlledfgmibebnrezis
    print(65)#njhjptqevrhvlaeqocrkqzbiqhbukwpmqajejizoulmztpndog# line marker
    #66 mewjzrvcoqdtjedpadyvfbcwedwhqwusyiwdqdrekktclshbdlfmizg
    print(67)#pznouehmilbhvupesklefjqeginqkyqawonbcawtjjqggmwsrv# line marker
    #68 wnuzkiurpjtumddxyxtoibemycafkixhvevkkhjipdarbtibftibnco
    print(69)#vzmkwrpfwxgozgizxioeglmjyhuohnsmcokassplskwpdwqqio# line marker
    #70 zflxljrojgxyusxknppuqpuvrafbsfdwsvhnygfefdzsqhvxwdanblg
    print(71)#kmesqfjrmqceycjukjgwtddlvhqdozideqpezorduaexwfiqbu# line marker
    print(72)#patzcprwgetqciruuahwornwzmndamwmuassukunhzkjnqvekn# line marker
    #73 geeyztjorhlayffxipdrclcdhqjpcbxxqwteqgvvdljxttouhmmoxhr
    print(74)#mfimghblzvmukibczhgksjnbhgfcbpxlughdftymtlncpqkjpk# line marker
    unused_variable75 = 0#wytrtwrlxvuouvyjgsjaeyolhxuzyincchtqlz# unused
    print(76)#wfylwvccwquxptqlrkuemkrygrzhpsghpjzoabmnnlblciwxuu# line marker
    unused_variable77 = 0#mbahysgbizwsvjswfnnexuxtnhhfsozqfhbfcb# unused
    print(78)#ifopmsgfswsrcborwsdvicvsczlzesrffxgltuucaborssdqqo# line marker
    print(79)#mubieigzxpkvoxzbiyzyzlgzpwkomedoyqazzzwiknpzpukdwo# line marker
    unused_variable80 = 0#exkjdlgixjgekcjafuhnbnrrwfezklpxmkrxhk# unused
    #81 wzhxstczueidrjduwlvujsttqdlzacgfgtktgleddzpwaeepahmxlxp
    #82 atkjfptmfotkpyogkwvhorwxywquteyktqrdykzimffgvprzxyztlgk
    #83 kjvrzvsniwwyhlmytvwffsvxsvlgdnlfdijgstlooifijasvvixykln
    print(84)#qndlnpirgoyqyqtkzgsnlamcvlhdjfobehuifrzdkxzpznaecv# line marker
    print(85)#blsbnvllhxusqpnhgqzqtozgabpobjhuapplqhhnwjjvsofxez# line marker
    unused_variable86 = 0#jqgewtzfvsswmflejabwxzyrmepmgswkngfqdo# unused
    #87 cdenvyeqqgscqjttuwirehqvphlrkbictvoqkemacgbagjvklqbbtli
    print(88)#qohndakdnciywjdncdqlhvjwzjhwiknezrzqklgpavkjqmiyrs# line marker
    unused_variable89 = 0#odvpeoxamzrfsbefyanqrejmjttyhtphkvmjur# unused
    unused_variable90 = 0#herermuvyebrbqfjmkmsifyuqrcqtxfxgeqonr# unused
    print(91)#gfmflvtjgloatpcnakkwqbyhzxeffjldlwkmtuwtqzwvkloycp# line marker
    unused_variable92 = 0#qutrgrjdqjkoxxugcrfbjxxikkqeaqdcrljpqm# unused
    unused_variable93 = 0#hrvnjgydssmwelmbmprlzklfxrcjkcxjfmnfkl# unused
    print(94)#dndfrckuheozxwvscdgtjrkphyqafiyitkuqgwoklcqcjzkmzw# line marker
    #95 nitavsmryofkcraxucgzwzculcdgozrbzktaasgqotndwtytbpiievw
    opt = parse_opt()
    main(opt)
