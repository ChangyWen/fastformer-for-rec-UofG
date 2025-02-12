# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
import logging
import random
from pathlib import Path
import numpy as np
import sys
import traceback

import torch
from tqdm.auto import tqdm
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from utility.utils import (setuplogger, init_process, cleanup_process, warmup_linear, get_device, lr_schedule,
                            get_barrier, only_on_main_process, check_args_environment, dump_args)
from utility.metrics import acc, MetricsDict
from parameters import parse_args

from data_handler.streaming import get_files
from data_handler.preprocess import get_news_feature, infer_news
from data_handler.TrainDataloader import DataLoaderTrainForSpeedyRec
from data_handler.TestDataloader import DataLoaderTest

from models.speedyrec import MLNR

from torch.utils.tensorboard import SummaryWriter
current_time = int(time.time())

def ddp_train_vd(args):
    '''
    Distributed training
    '''
    setuplogger()
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    args = check_args_environment(args)

    logging.info('-----------start train------------')

    cache_state = mp.Manager().dict()
    data_files = mp.Manager().list([])
    end_dataloder = mp.Manager().Value('b', False)
    end_train = mp.Manager().Value('b', False)
    mp.spawn(train,
             args=(
             args, cache_state, data_files, end_dataloder, end_train),
             nprocs=args.world_size,
             join=True)


def train(local_rank,
          args,
          cache_state,
          data_files,
          end_dataloder,
          end_train,
          dist_training=True):


    setuplogger()
    summary_writer = SummaryWriter(log_dir=f'./tensorboard_logs/{current_time}/local_rank_{local_rank}')
    try:
        if dist_training:
            init_process(local_rank, args.world_size)
        device = get_device()
        barrier = get_barrier(dist_training)

        news_info, news_combined = get_news_feature(args, mode='train')
        with only_on_main_process(local_rank, barrier) as need:
            if need:
                data_paths = []
                data_dirs = os.path.join(args.root_data_dir, 'train/')
                data_paths.extend(get_files(data_dirs, args.filename_pat))
                data_paths.sort()

        model = MLNR(args)
        if args.load_ckpt_name.endswith('.pt'):
            # train_path = os.path.join(args.pretrained_model_path, 'fastformer4rec.pt')
            model.load_param(args.load_ckpt_name)
            print(f'****** Load checkpoint {args.load_ckpt_name} successfully ******')


        model = model.to(device)
        rest_param = filter(
            lambda x: id(x) not in list(map(id, model.news_encoder.unicoder.parameters())),
            model.parameters())
        optimizer = optim.Adam([{
            'params': model.news_encoder.unicoder.parameters(),
            'lr': args.pretrain_lr  #lr_schedule(args.pretrain_lr, 1, args)
        }, {
            'params': rest_param,
            'lr': args.lr  #lr_schedule(args.lr, 1, args)
        }])
        #

        if dist_training:
            ddp_model = DDP(model,
                            device_ids=[local_rank],
                            output_device=local_rank,
                            find_unused_parameters=True,
                            broadcast_buffers=False)
        else:
            ddp_model = model

        logging.info('Training...')
        start_time = time.time()
        test_time = 0.0
        global_step = 0
        best_count = 0
        optimizer.zero_grad()

        loss = 0.0
        best_auc = 0.0
        accuary = 0.0
        hit_num = 0
        all_num = 1
        encode_num = 0
        cache = np.zeros((len(news_combined),args.news_dim))
        for ep in range(args.epochs):
            with only_on_main_process(local_rank, barrier) as need:
                if need:
                    while len(data_files) > 0:
                        data_files.pop()
                    data_files.extend(data_paths)
                    random.shuffle(data_files)
            barrier()

            dataloader = DataLoaderTrainForSpeedyRec(
                args=args,
                data_files=data_files,
                cache_state=cache_state,
                end=end_dataloder,
                local_rank=local_rank,
                world_size=args.world_size,
                news_features=news_combined,
                news_index=news_info.news_index,
                enable_prefetch=args.enable_prefetch,
                enable_prefetch_stream=args.enable_prefetch_stream,
                global_step=global_step,
                add_pad_news=True)

            ddp_model.train()
            pad_doc = torch.zeros(1, args.news_dim, device=device)

            for cnt, batch in tqdm(enumerate(dataloader)):
                with torch.autograd.set_detect_anomaly(False):
                    address_cache, update_cache, satrt_inx, end_inx, batch = batch
                    global_step += 1

                    if args.enable_gpu:
                        input_ids, hist_sequence, hist_sequence_mask, candidate_inx, label_batch = (
                            x.cuda(device=device,non_blocking=True) if x is not None else x
                            for x in batch[:5])
                    else:
                        input_ids, hist_sequence, hist_sequence_mask, candidate_inx, label_batch = batch[:5]

                    encode_num += input_ids.size(0)

                    # Get news vecs from cache.
                    if address_cache is not None:
                        # cache_vec = [cache[inx] for inx in address_cache]
                        cache_vec = cache[address_cache]
                        cache_vec = torch.FloatTensor(
                            cache_vec).cuda(device=device, non_blocking=True)

                        # atime += time.time() - temp_stime
                        hit_num += cache_vec.size(0)
                        all_num += cache_vec.size(0)

                    else:
                        cache_vec = None
                        hit_num += 0

                    if cache_vec is not None:
                        cache_vec = torch.cat([pad_doc, cache_vec], 0)
                    else:
                        cache_vec = pad_doc

                    if input_ids.size(0) > 0:
                        if dist_training:
                            encode_vecs = ddp_model.module.news_encoder(input_ids)
                        else:
                            encode_vecs = ddp_model.news_encoder(input_ids)
                    else:
                        encode_vecs = None

                    all_tensors = [torch.empty_like(encode_vecs) for _ in range(args.world_size)]
                    dist.all_gather(all_tensors, encode_vecs)
                    all_tensors[local_rank] = encode_vecs
                    all_encode_vecs = torch.cat(all_tensors, dim=0)
                    news_vecs = torch.cat([cache_vec, all_encode_vecs], 0)

                    all_num += all_encode_vecs.size(0)
                    bz_loss, y_hat = ddp_model(news_vecs,
                                             hist_sequence, hist_sequence_mask,
                                             candidate_inx,
                                             label_batch)

                    loss += bz_loss.item()
                    assert not torch.isnan(bz_loss).any(), f'nan loss: {bz_loss}'
                    bz_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    current_acc = acc(label_batch, y_hat)
                    accuary += current_acc

                    summary_writer.add_scalar('loss', bz_loss, global_step)
                    summary_writer.add_scalar('acc', current_acc, global_step)

                    # update the cache
                    if args.max_step_in_cache > 0 and encode_vecs is not None:
                        update_vecs = all_encode_vecs.detach().cpu().numpy()[:len(update_cache)]
                        cache[update_cache] = update_vecs

                    optimizer.param_groups[0]['lr'] = lr_schedule(args.pretrain_lr, global_step, args)
                    optimizer.param_groups[1]['lr'] = lr_schedule(args.lr, global_step, args)

                    barrier()

                if global_step % args.log_steps == 0:
                    logging.info(
                        '[{}] cost_time:{} step:{}, train_loss: {:.5f}, acc:{:.5f}, hit:{}, encode_num:{}, lr:{:.8f}, pretrain_lr:{:.8f}'.format(
                            local_rank, time.time() - start_time-test_time, global_step, loss / args.log_steps, accuary / args.log_steps, hit_num/all_num, encode_num,
                            optimizer.param_groups[1]['lr'], optimizer.param_groups[0]['lr']))
                    loss = 0.0
                    accuary = 0.0

                # if global_step%args.test_steps == 0 and local_rank == 0:
                #     stest_time = time.time()
                #     auc = test(model, args, device, news_info.category_dict, news_info.subcategory_dict)
                #     ddp_model.train()
                #     logging.info('step:{}, auc:{}'.format(global_step, auc))
                #     test_time = test_time + time.time()-stest_time

                # save model minibatch
                if local_rank == 0 and global_step % args.save_steps == 0:
                    ckpt_path = os.path.join(args.model_dir, f'{args.savename}-epoch-{ep + 1}-{global_step}.pt')
                    torch.save(
                        {
                            'model_state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'category_dict': news_info.category_dict,
                            'subcategory_dict': news_info.subcategory_dict,
                        }, ckpt_path)
                    logging.info(f"Model saved to {ckpt_path}")

            logging.info('epoch:{}, time:{}, encode_num:{}'.format(ep + 1, time.time() - start_time-test_time, encode_num))
            # save model after an epoch
            if local_rank == 0:
                ckpt_path = os.path.join(args.model_dir, '{}-epoch-{}.pt'.format(args.savename, ep + 1))
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'category_dict': news_info.category_dict,
                        'subcategory_dict': news_info.subcategory_dict,
                    }, ckpt_path)
                logging.info(f"Model saved to {ckpt_path}")

                # auc = test(model, args, device, news_info.category_dict, news_info.subcategory_dict)
                # ddp_model.train()

                # if auc>best_auc:
                #     best_auc = auc
                # else:
                #     best_count += 1
                #     if best_auc >= 3:
                #         logging.info("best_auc:{}, best_ep:{}".format(best_auc, ep-3))
                #         end_train.value = True
            barrier()
            # if end_train.value:
            #     break

        if dist_training:
            cleanup_process()

        summary_writer.flush()
        summary_writer.close()

    except:
        error_type, error_value, error_trace = sys.exc_info()
        traceback.print_tb(error_trace)
        logging.info(error_value)



def test(model, args, device, category_dict, subcategory_dict):
    model.eval()

    with torch.no_grad():
        news_info, news_combined = get_news_feature(args, mode='dev', category_dict=category_dict, subcategory_dict=subcategory_dict)
        news_vecs = infer_news(model, device, news_combined)

        dataloader = DataLoaderTest(
            news_index=news_info.news_index,
            news_scoring=news_vecs,
            data_dirs=[os.path.join(args.root_data_dir,
                                    f'dev/')],
            filename_pat=args.filename_pat,
            args=args,
            world_size=1,
            worker_rank=0,
            cuda_device_idx=0,
            enable_prefetch=args.enable_prefetch,
            enable_shuffle=args.enable_shuffle,
            enable_gpu=args.enable_gpu,
        )

        results = MetricsDict(metrics_name=["AUC", "MRR", "nDCG5", "nDCG10"])
        results.add_metric_dict('all users')
        results.add_metric_dict('cold users')

        for cnt, (log_vecs, log_mask, news_vecs, labels) in enumerate(dataloader):
            his_lens = torch.sum(log_mask, dim=-1).to(torch.device("cpu")).detach().numpy()

            if args.enable_gpu:
                log_vecs = log_vecs.cuda(device=device, non_blocking=True)
                log_mask = log_mask.cuda(device=device, non_blocking=True)

            user_vecs = model.user_encoder(
                log_vecs, log_mask, user_log_mask=True).to(torch.device("cpu")).detach().numpy()

            for index, user_vec, news_vec, label, his_len in zip(
                    range(len(labels)), user_vecs, news_vecs, labels, his_lens):

                if label.mean() == 0 or label.mean() == 1:
                    continue
                score = np.dot(
                    news_vec, user_vec
                )

                metric_rslt = results.cal_metrics(score, label)
                results.update_metric_dict('all users', metric_rslt)

                if his_len <= 5:
                    results.update_metric_dict('cold users', metric_rslt)

            # if cnt % args.log_steps == 0:
            #     results.print_metrics(0, cnt * args.batch_size, 'all users')
            #     results.print_metrics(0, cnt * args.batch_size, 'cold users')

        dataloader.join()
        for i in range(2):
            results.print_metrics(0, cnt * args.batch_size, 'all users')
            results.print_metrics(0, cnt * args.batch_size, 'cold users')

    return np.mean(results.metrics_dict["all users"]['AUC'])


if __name__ == '__main__':
    setuplogger()
    args = parse_args()
    ddp_train_vd(args)



# nohup \
# python train.py \
# --pretreained_model others \
# --pretrained_model_path google-bert/bert-base-uncased \
# --do_lower_case True \
# --root_data_dir ./data/speedy_data/ \
# --num_hidden_layers 8 \
# --world_size 2 \
# --lr 1e-4 \
# --pretrain_lr 8e-6 \
# --warmup True \
# --schedule_step 240000 \
# --warmup_step 1000 \
# --batch_size 16 \
# --npratio 4 \
# --beta_for_cache 0.002 \
# --max_step_in_cache 2 \
# --savename speedyrec_mind \
# --news_dim 256 \
# --save_steps 28000 \
# >./nohup_logs/1.out 2>&1 &


# AutoDL:
# [INFO 2025-02-08 14:16:49,446] epoch:1, time:52431.833335876465, encode_num:24875712
# [INFO 2025-02-08 14:16:52,450] Model saved to ./saved_models/speedyrec_mind-epoch-1.pt

# python train.py \
# --pretreained_model others \
# --pretrained_model_path google-bert/bert-base-uncased \
# --do_lower_case True \
# --load_ckpt_name ./saved_models/speedyrec_mind-epoch-1.pt \
# --root_data_dir ./data/speedy_data/ \
# --num_hidden_layers 8 \
# --world_size 2 \
# --lr 1e-4 \
# --pretrain_lr 8e-6 \
# --warmup True \
# --schedule_step 240000 \
# --warmup_step 1000 \
# --batch_size 16 \
# --npratio 4 \
# --beta_for_cache 0.002 \
# --max_step_in_cache 2 \
# --savename speedyrec_mind-1 \
# --news_dim 256 \
# --save_steps 28000


###### continued training of roberta-base epoch 3 on HKU server
# python train.py \
# --pretreained_model others \
# --pretrained_model_path FacebookAI/roberta-base \
# --do_lower_case False \
# --load_ckpt_name ./saved_models/autodl-speedyrec_mind_roberta-epoch-3.pt \
# --savename speedyrec_mind_roberta-3 \
# --lr 5e-5 \
# --pretrain_lr 4e-6 \
# --world_size 2 \
# --batch_size 16 \
# --root_data_dir ./data/speedy_data/ \
# --num_hidden_layers 8 \
# --warmup True \
# --schedule_step 240000 \
# --warmup_step 1000 \
# --npratio 4 \
# --beta_for_cache 0.002 \
# --max_step_in_cache 2 \
# --news_dim 256 \
# --save_steps 50000



# 103859it [13:03:28,  2.23it/s][INFO 2025-02-10 00:22:18,286] [1] cost_time:94915.46762371063 step:209600, train_loss: 1.16156, acc:0.55312, hit:0.14958152740649958, encode_num:24695097, lr:0.00001272, pretrain_lr:0.00000102
# [INFO 2025-02-10 00:22:18,286] [0] cost_time:94915.46768951416 step:209600, train_loss: 1.16156, acc:0.55312, hit:0.14958152740649958, encode_num:24695097, lr:0.00001272, pretrain_lr:0.00000102
# 104059it [13:04:59,  2.15it/s][INFO 2025-02-10 00:23:49,598] [1] cost_time:95006.77956151962 step:209800, train_loss: 1.18220, acc:0.54828, hit:0.14957973927450333, encode_num:24718976, lr:0.00001264, pretrain_lr:0.00000101
# 104060it [13:05:00,  2.11it/s][INFO 2025-02-10 00:23:49,598] [0] cost_time:95006.77962064743 step:209800, train_loss: 1.18220, acc:0.54828, hit:0.14957973927450333, encode_num:24718976, lr:0.00001264, pretrain_lr:0.00000101
# 104259it [13:06:29,  2.18it/s][INFO 2025-02-10 00:25:19,829] [1] cost_time:95097.01033878326 step:210000, train_loss: 1.19834, acc:0.52953, hit:0.14958023516894317, encode_num:24742491, lr:0.00001255, pretrain_lr:0.00000100
# 104260it [13:06:30,  2.32it/s][INFO 2025-02-10 00:25:19,829] [0] cost_time:95097.01057267189 step:210000, train_loss: 1.19834, acc:0.52953, hit:0.14958023516894317, encode_num:24742491, lr:0.00001255, pretrain_lr:0.00000100
# 104459it [13:08:01,  2.23it/s][INFO 2025-02-10 00:26:51,277] [0] cost_time:95188.45823192596 step:210200, train_loss: 1.19131, acc:0.53484, hit:0.14957833501498913, encode_num:24766478, lr:0.00001247, pretrain_lr:0.00000100
# 104460it [13:08:01,  2.26it/s][INFO 2025-02-10 00:26:51,277] [1] cost_time:95188.4584941864 step:210200, train_loss: 1.19131, acc:0.53484, hit:0.14957833501498913, encode_num:24766478, lr:0.00001247, pretrain_lr:0.00000100
# 104659it [13:09:33,  2.37it/s][INFO 2025-02-10 00:28:23,058] [1] cost_time:95280.2399122715 step:210400, train_loss: 1.18105, acc:0.54750, hit:0.1495804816755464, encode_num:24790166, lr:0.00001238, pretrain_lr:0.00000099
# 104660it [13:09:33,  2.36it/s][INFO 2025-02-10 00:28:23,059] [0] cost_time:95280.2400097847 step:210400, train_loss: 1.18105, acc:0.54750, hit:0.1495804816755464, encode_num:24790166, lr:0.00001238, pretrain_lr:0.00000099
# 104859it [13:11:04,  2.28it/s][INFO 2025-02-10 00:29:54,140] [1] cost_time:95371.32122612 step:210600, train_loss: 1.18992, acc:0.53562, hit:0.1495802045762094, encode_num:24813820, lr:0.00001230, pretrain_lr:0.00000098
# 104860it [13:11:04,  2.30it/s][INFO 2025-02-10 00:29:54,140] [0] cost_time:95371.32130789757 step:210600, train_loss: 1.18992, acc:0.53562, hit:0.1495802045762094, encode_num:24813820, lr:0.00001230, pretrain_lr:0.00000098
# 105059it [13:12:34,  2.24it/s][INFO 2025-02-10 00:31:24,488] [0] cost_time:95461.66933131218 step:210800, train_loss: 1.20671, acc:0.53016, hit:0.14957882415902218, encode_num:24837175, lr:0.00001222, pretrain_lr:0.00000098
# 105060it [13:12:34,  2.21it/s][INFO 2025-02-10 00:31:24,488] [1] cost_time:95461.66960549355 step:210800, train_loss: 1.20671, acc:0.53016, hit:0.14957882415902218, encode_num:24837175, lr:0.00001222, pretrain_lr:0.00000098
# 105259it [13:14:04,  2.25it/s][INFO 2025-02-10 00:32:54,083] [1] cost_time:95551.2642197609 step:211000, train_loss: 1.19014, acc:0.54625, hit:0.149582603464868, encode_num:24860500, lr:0.00001213, pretrain_lr:0.00000097
# 105260it [13:14:04,  2.49it/s][INFO 2025-02-10 00:32:54,083] [0] cost_time:95551.26428937912 step:211000, train_loss: 1.19014, acc:0.54625, hit:0.149582603464868, encode_num:24860500, lr:0.00001213, pretrain_lr:0.00000097
# 105459it [13:15:33,  2.14it/s][INFO 2025-02-10 00:34:23,207] [1] cost_time:95640.38874816895 step:211200, train_loss: 1.18015, acc:0.55266, hit:0.1495814156476318, encode_num:24883698, lr:0.00001205, pretrain_lr:0.00000096
# [INFO 2025-02-10 00:34:23,207] [0] cost_time:95640.38882350922 step:211200, train_loss: 1.18015, acc:0.55266, hit:0.1495814156476318, encode_num:24883698, lr:0.00001205, pretrain_lr:0.00000096
# 105659it [13:17:12,  1.96it/s][INFO 2025-02-10 00:36:02,439] [1] cost_time:95739.62082242966 step:211400, train_loss: 1.20112, acc:0.54062, hit:0.14958394781895762, encode_num:24909619, lr:0.00001197, pretrain_lr:0.00000096
# 105660it [13:17:12,  1.94it/s][INFO 2025-02-10 00:36:02,440] [0] cost_time:95739.62093734741 step:211400, train_loss: 1.20112, acc:0.54062, hit:0.14958394781895762, encode_num:24909619, lr:0.00001197, pretrain_lr:0.00000096
# 105734it [13:17:48,  2.53it/s]2025-02-10 00:36:38.387765: I tensorflow/core/framework/local_rendezvous.cc:405] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
#          [[{{node IteratorGetNext}}]]
# 2025-02-10 00:36:38.387778: I tensorflow/core/framework/local_rendezvous.cc:405] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
#          [[{{node IteratorGetNext}}]]
# 105740it [13:17:50,  2.21it/s]
# 105740it [13:17:50,  2.21it/s]
# [INFO 2025-02-10 00:36:40,495] epoch:2, time:95777.67623448372, encode_num:24919533
# [INFO 2025-02-10 00:36:40,495] epoch:2, time:95777.67642903328, encode_num:24919533
# [INFO 2025-02-10 00:36:43,704] Model saved to ./saved_models/speedyrec_mind-1-epoch-2.pt
# [INFO 2025-02-10 00:36:44,110] start async...
# [INFO 2025-02-10 00:36:44,111] start async...
# [INFO 2025-02-10 00:36:44,111] visible_devices:[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
# [INFO 2025-02-10 00:36:44,112] visible_devices:[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
# 119it [00:53,  2.42it/s][INFO 2025-02-10 00:37:38,219] [0] cost_time:95835.40071725845 step:211600, train_loss: 1.17971, acc:0.55141, hit:0.14958857588104177, encode_num:24933380, lr:0.00001188, pretrain_lr:0.00000095
# [INFO 2025-02-10 00:37:38,219] [1] cost_time:95835.40091633797 step:211600, train_loss: 1.17971, acc:0.55141, hit:0.14958857588104177, encode_num:24933380, lr:0.00001188, pretrain_lr:0.00000095
# 319it [02:26,  2.03it/s][INFO 2025-02-10 00:39:10,522] [0] cost_time:95927.70322203636 step:211800, train_loss: 1.20354, acc:0.54031, hit:0.14958971596344386, encode_num:24956823, lr:0.00001180, pretrain_lr:0.00000094
# [INFO 2025-02-10 00:39:10,522] [1] cost_time:95927.70334720612 step:211800, train_loss: 1.20354, acc:0.54031, hit:0.14958971596344386, encode_num:24956823, lr:0.00001180, pretrain_lr:0.00000094
# 482it [03:40,  2.19it/s]^C



# [INFO 2025-02-10 14:41:47,208] visible_devices:[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
# 59it [00:27,  2.28it/s][INFO 2025-02-10 14:42:14,698] [1] cost_time:47807.56764173508 step:105800, train_loss: 1.15445, acc:0.56719, hit:0.14930141749212877, encode_num:12471430, lr:0.00005615, pretrain_lr:0.00000449
# 60it [00:27,  2.32it/s][INFO 2025-02-10 14:42:14,699] [0] cost_time:47807.57531237602 step:105800, train_loss: 1.15445, acc:0.56719, hit:0.14930141749212877, encode_num:12471430, lr:0.00005615, pretrain_lr:0.00000449
# 259it [01:57,  2.03it/s][INFO 2025-02-10 14:43:44,821] [1] cost_time:47897.68967676163 step:106000, train_loss: 1.18504, acc:0.55437, hit:0.1493104690240422, encode_num:12494519, lr:0.00005607, pretrain_lr:0.00000449
# [INFO 2025-02-10 14:43:44,821] [0] cost_time:47897.69729924202 step:106000, train_loss: 1.18504, acc:0.55437, hit:0.1493104690240422, encode_num:12494519, lr:0.00005607, pretrain_lr:0.00000449
# 459it [03:29,  2.24it/s][INFO 2025-02-10 14:45:17,024] [0] cost_time:47989.900277376175 step:106200, train_loss: 1.18601, acc:0.55078, hit:0.1493118303515531, encode_num:12518072, lr:0.00005598, pretrain_lr:0.00000448
# 460it [03:29,  2.37it/s][INFO 2025-02-10 14:45:17,024] [1] cost_time:47989.89297533035 step:106200, train_loss: 1.18601, acc:0.55078, hit:0.1493118303515531, encode_num:12518072, lr:0.00005598, pretrain_lr:0.00000448
# 659it [04:59,  2.24it/s][INFO 2025-02-10 14:46:46,816] [0] cost_time:48079.69296836853 step:106400, train_loss: 1.18701, acc:0.55016, hit:0.1493138065984874, encode_num:12541202, lr:0.00005590, pretrain_lr:0.00000447
# [INFO 2025-02-10 14:46:46,817] [1] cost_time:48079.68564367294 step:106400, train_loss: 1.18701, acc:0.55016, hit:0.1493138065984874, encode_num:12541202, lr:0.00005590, pretrain_lr:0.00000447