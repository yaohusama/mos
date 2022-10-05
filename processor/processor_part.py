import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist

def do_train(cfg,
             model,
             train_loader,
             val_loader,
             optimizer,
             scheduler,
             loss_fn,
             num_query, local_rank):
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(cfg.OUTPUT_DIR)  # 数据存放在这个文件夹
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.trainer")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    # if device:
    #     model.to(local_rank)
    #     if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
    #         print('Using {} GPUs for training'.format(torch.cuda.device_count()))
    #         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # trainer
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        if cfg.MODEL.NAME == 'resnet50_ibn_a':
            # scheduler.step()
            scheduler.step(epoch)
        else:
            scheduler.step(epoch)

        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            # optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                if cfg.MODEL.NAME =='resnet50_ibn_a':
                    score, feat, part_score = model(img, target)
                else:
                    score, feat, part_score = model(img, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, part_score, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            # if 'center' in cfg.MODEL.METRIC_LOSS_TYPE: #wo
            #     for param in center_criterion.parameters():
            #         param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
            #     scaler.step(optimizer_center)
            #     scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                if cfg.MODEL.NAME == 'resnet50_ibn_a' or cfg.MODEL.NAME == 'resnet50':
                    # logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                    #             .format(epoch, (n_iter + 1), len(train_loader),
                    #                     loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                    # writer.add_scalar('Loss', loss_meter.avg,
                    #                   global_step=len(train_loader) * epoch + n_iter)
                    # writer.add_scalar('acc', acc_meter.avg,
                    #                   global_step=len(train_loader) * epoch + n_iter)
                    # writer.add_scalar('Lr', scheduler._get_lr(epoch)[0], global_step=len(train_loader) * epoch + n_iter)
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
                    writer.add_scalar('Loss', loss_meter.avg,
                                      global_step=len(train_loader) * epoch + n_iter)
                    writer.add_scalar('acc', acc_meter.avg,
                                      global_step=len(train_loader) * epoch + n_iter)
                    writer.add_scalar('Lr', scheduler.get_lr()[0], global_step=len(train_loader) * epoch + n_iter)
                else:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                    writer.add_scalar('Loss', loss_meter.avg,
                                      global_step=len(train_loader) * epoch + n_iter)
                    writer.add_scalar('acc', acc_meter.avg,
                                      global_step=len(train_loader) * epoch + n_iter)
                    writer.add_scalar('Lr', scheduler._get_lr(epoch)[0], global_step=len(train_loader) * epoch + n_iter)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        if cfg.MODEL.NAME == 'resnet50_ibn_a':
                            feat = model(img)
                        else:
                            feat = model(img, cam_label=camids, view_label=target_view)

                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
                
            writer.add_scalar('mAP', mAP, global_step=epoch)

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    # print('Total running time: ', timedelta(seconds=all_end_time - all_start_time))
    logger.info("Total running time: {}".format(total_time))


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


