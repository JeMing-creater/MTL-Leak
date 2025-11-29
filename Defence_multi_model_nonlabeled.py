import os
import cv2
import yaml
import torch
import datetime
from torch import nn
import numpy as np
from objprint import objstr
from termcolor import colored
from easydict import EasyDict
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from accelerate import Accelerator
from utils.config import create_config
from timm.optim import optim_factory
from utils.utils import AverageMeter, ProgressMeter, PerformanceMeter
from utils.common_config import get_train_dataset, get_transformations, \
    get_val_dataset, get_train_dataloader, get_val_dataloader, \
    get_optimizer, get_model, adjust_learning_rate, \
    get_criterion
from utils.evaluate_utils import validate_results, calculate_multi_task_performance, eval_all_results, get_output, \
    get_output, SemsegMeter, DepthMeter, SaliencyMeter
from utils.utils import Logger
from utils.resum import load_pth
from utils.loss_functions import get_loss_meters
from Multi_data_division import division
from utils.optimizer import LinearWarmupCosineAnnealingLR
from files.models.classification_model import ClassFormer
from utils.MIAutils import get_resnet_classification_model
import pickle
import random
from torch import optim
import csv
import matplotlib.pyplot as plt
import math

# 配置读取
args = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))  # 读取配置
cv2.setNumThreads(0)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.trainer.card_ID



def give_target(pred, task='semseg'):
    if task == 'semseg' or task == 'human_parts':
        pred = pred.permute(0, 2, 3, 1)
        _, pred = torch.max(pred, dim=3)
        return pred.unsqueeze(1)
    elif task == 'depth':
        return pred
    elif task == 'normals':
        pred = pred.permute(0, 2, 3, 1)
        pred = (F.normalize(pred, p=2, dim=3) + 1.0) * 255 / 2.0
        pred = 2 * pred / 255 - 1
        pred = pred.permute(0, 3, 1, 2)  # [B, C, H, W]
        return pred
    elif task == 'sal':
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.squeeze(255 * 1 / (1 + torch.exp(-pred)))
        pred = pred.float().squeeze() / 255.

        s = SaliencyMeter()
        mask_thres = s.mask_thres
        mask_eval = None
        for j, thres in enumerate(mask_thres):
            if mask_eval == None:
                mask_eval = (pred > thres)
            else:
                mask_eval += (pred > thres)
        return mask_eval.float().unsqueeze(1)


def task_Similar_calculations(p, model, batch, losses, criterion, performance_meter, segmentation_altho=0.2,
                              similar_altho=0.5, if_once_epoch=False):
    def batch_segmentation(batch, segmentation_altho=0.2):
        batch_num = batch['image'].size(0)
        indices = torch.randperm(batch_num)
        for key in batch.keys():
            if key == 'meta':
                continue
            else:
                batch[key] = batch[key][indices]
        task_batch = {}
        non_task_batch = {}
        non_task_num = int(segmentation_altho * batch_num)
        if non_task_num == 0:
            non_task_num = 1
        if non_task_num == batch_num:
            non_task_num = batch_num - 1
        elif batch_num == 1:
            for key in batch.keys():
                non_task_batch[key] = batch[key]
                task_batch[key] = batch[key].clone()
                # non_task_guide_batch[key] = batch[key].clone()
            return task_batch, non_task_batch
        for key in batch.keys():
            if key == 'meta':
                continue
            non_task_batch[key] = batch[key][0:non_task_num]
            task_batch[key] = batch[key][non_task_num:]
            # non_task_guide_batch[key] = batch[key][non_task_num:non_task_num + non_task_num].clone()

        return task_batch, non_task_batch

    def tasks_performance(p, model, batch, losses, criterion, performance_meter):
        images = batch['image'].to(accelerator.device)
        targets = {task: batch[task].detach().to(accelerator.device) for task in p.ALL_TASKS.NAMES}
        output = model(images)

        # task_loss
        loss_dict = criterion(output, targets)
        for k, v in loss_dict.items():
            losses[k].update(v.item())
        performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES},
                                 {t: targets[t] for t in p.TASKS.NAMES})

        return loss_dict['total']

    def similar_calculations(p, model, batch, guide_batch, losses, criterion, performance_meter):
        def test_acc(p, tasks, guide_tasks):
            targets = {task: give_target(guide_tasks[task].detach().to(accelerator.device), task) for task in
                       p.ALL_TASKS.NAMES}
            # task_loss
            loss_dict = criterion(tasks, targets)

            return loss_dict['total']

        # images = batch['image'].to(accelerator.device)
        # guide_task = model(images)
        # for key in guide_task.keys():
        #     if key == 'deep_supervision':
        #         continue
        #     guide_task[key] = guide_task[key].detach()
        # mem_task = model(images)
        # return test_acc(p, mem_task, guide_task)
        images = batch['image'].to(accelerator.device)
        guide_images = guide_batch['image'].to(accelerator.device)
        
        target  = {task: batch[task].to(accelerator.device) for task in p.ALL_TASKS.NAMES}
        guide_target = {task: guide_batch[task].to(accelerator.device) for task in p.ALL_TASKS.NAMES}
        
        guide_out = model(guide_images)
        out = model(images)
        loss1 = criterion(out, target)['total'] 
        loss2 = criterion(guide_out, guide_target)['total'] 
        if loss1 < loss2:
            return torch.tensor(0)
        else:
            loss = loss1 - loss2
            return loss

    task_batch, non_task_batch = batch_segmentation(batch, segmentation_altho)
    task_loss = tasks_performance(p, model, task_batch, losses, criterion, performance_meter)
    similar_loss = 0
    if if_once_epoch == True:
        all_loss = task_loss
    else:
        similar_loss = similar_calculations(p, model, non_task_batch, task_batch, losses, criterion,
                                            performance_meter)
        all_loss = task_loss + similar_altho * similar_loss
    return all_loss,task_loss,similar_altho * similar_loss


def defence_train_model(p, model, criterion, optimizer, data, start_epoch, best_result, accelerator,
                        segmentation_altho=0.2, similar_altho=0.5, shadow=False,if_once_epoch=1):
    train, test = data
    if shadow == False:
        shadow = 'Target'
    else:
        shadow = 'Shadow'
    for epoch in range(start_epoch, p['epochs']):
        # lr加载不受p指导影响
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        losses = get_loss_meters(p)
        performance_meter = PerformanceMeter(p)
        progress = ProgressMeter(len(train),
                                 [v for v in losses.values()],
                                 prefix="{} Epoch: [{}/{}]".format(shadow, epoch, p['epochs']))
        model.train()
        i = 0
        if_once = False
        if epoch < if_once_epoch:
            if_once = True
        a_task_loss = 0
        a_similar_loss = 0
        batch_num = 0
        for batch in train:
            all_loss,task_loss,similar_loss = task_Similar_calculations(p=p, model=model, batch=batch, losses=losses, criterion=criterion,
                                                 performance_meter=performance_meter, if_once_epoch=if_once,
                                                 segmentation_altho=segmentation_altho, similar_altho=similar_altho)
            a_task_loss += task_loss.item()
            if if_once:
                a_similar_loss += similar_loss
            else:
                a_similar_loss += similar_loss.item()
            batch_num += 1
            # Backward
            optimizer.zero_grad()
            accelerator.backward(all_loss)
            # loss_dict['total'].backward()
            optimizer.step()

            if i % 25 == 0:
                progress.display(i)
            i += 1
        a_task_loss = a_task_loss/batch_num
        a_similar_loss = a_similar_loss/batch_num
        accelerator.print('{} Epoch: [{}/{}] a_task_loss: {} a_similar_loss: {}'.format(shadow,epoch,p['epochs'],a_task_loss,a_similar_loss))
        
        # Evaluate
        # 此处本该为最后轮作验证，但由于验证过于耗时，改为最后10轮验证
        eval_bool = False
        if p['10_eval'] == True:
            if (epoch + 1) % 10 == 0 or epoch + 1 > p['epochs'] - 10:
                eval_bool = True
            else:
                eval_bool = False
        else:
            eval_bool = True
        # eval_bool = True
        # Perform evaluation
        if eval_bool:
            print('Evaluate ...')
            # save_model_predictions(p, test, model, accelerator)
            _ = eval_all_results(p, model, train, accelerator)
            curr_result = eval_all_results(p, model, test, accelerator)
            improves, best_result = validate_results(p, curr_result, best_result)
            if improves:
                print('Save new best model')
                torch.save(model.state_dict(), p['best_model'])
        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1, 'best_result': best_result}, p['checkpoint'])
    return model


def eval_acc(p, model, data, accelerator, shadow=False):
    train, test = data
    if shadow == False:
        shadow = 'Target'
    else:
        shadow = 'Shadow'
    # model_checkpoint = torch.load(p['checkpoint'])['model']
    try:
        model_checkpoint = torch.load(p['best_model'])
    except:
        model_checkpoint = torch.load(p['checkpoint'])['model']
    model_checkpoint = load_pth(p, model_checkpoint)
    model.load_state_dict(model_checkpoint)
    # save_model_predictions(p, train, model, accelerator)
    print('eval Train')
    print(colored('Evaluating best {} model in train'.format(shadow), 'blue'))
    train_eval_stats = eval_all_results(p, model, train, accelerator)
    print('eval Test')
    print(colored('Evaluating best {} model in test'.format(shadow), 'blue'))
    # save_model_predictions(p, test, model, accelerator)
    test_eval_stats = eval_all_results(p, model, test, accelerator)
    return train_eval_stats, test_eval_stats


def give_config_exp(dataset, model_choose):
    if dataset == 'nyud':
        exp = './files/models/yml/nyud/'
        if 'hrnet18' in model_choose:
            exp = exp + 'hrnet18/'
        else:
            exp = exp + 'resnet50/'
    else:
        exp = './files/models/yml/pascal/'
        if 'hrnet18' in model_choose:
            exp = exp + 'hrnet18/'
        else:
            exp = exp + 'resnet18/'
    if 'MTL_Baseline' in model_choose:
        exp = exp + 'multi_task_baseline.yml'
    elif 'PADNet' in model_choose:
        exp = exp + 'pad_net.yml'
    elif 'MTINet' in model_choose:
        exp = exp + 'mti_net.yml'
    elif 'MTAN' in model_choose:
        exp = exp + 'mtan.yml'
    elif 'CrossStitch' in model_choose:
        exp = exp + 'cross_stitch.yml'
    elif 'NDDRCNN' in model_choose:
        exp = exp + 'nddr_cnn.yml'
    return exp


def resum(p, model, test, accelerator):
    if p['resum'] != True:
        accelerator.print(colored('remove checkpoint', 'blue'))
        if os.path.exists(p['checkpoint']):
            os.remove(p['checkpoint'])
    # Resume from checkpoint
    if os.path.exists(p['checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['checkpoint']), 'blue'))
        checkpoint = torch.load(p['checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model_checkpoint = load_pth(p, checkpoint['model'])
        model.load_state_dict(model_checkpoint)
        start_epoch = checkpoint['epoch']
        best_result = checkpoint['best_result']
    else:
        print(colored('No checkpoint file at {}'.format(p['checkpoint']), 'blue'))
        start_epoch = 0
        best_result = eval_all_results(p, model, test, accelerator)
    return model, start_epoch, best_result


def give_label_data(p, model, dataset, accelerator, save_dir, shadow=False):
    def save_data_to_local(data, label, file_path):
        # 将数据和标签存储到元组中
        data_tuple = (data, label)
        # 将元组序列化为字节流
        data_bytes = pickle.dumps(data_tuple)
        # 将字节流写入到文件
        with open(file_path, 'wb') as f:
            f.write(data_bytes)
        print('label data has save in: ', file_path)

    train, test = dataset
    model.eval()
    attack_x = []
    attack_y = []
    if len(train) > len(test):
        keep_num = len(test)
    else:
        keep_num = len(train)
    num = 0
    for batch in train:
        if num >= keep_num:
            break
        le = int(batch['image'].shape[0] / 2)
        images = batch['image'][0:le].to(accelerator.device)
        output = model(images)
        new_dict = {task: output[task].detach().cpu().numpy() for task in p.ALL_TASKS.NAMES}
        label = np.ones(batch['image'].shape[0])
        attack_x.append(new_dict)
        attack_y.append(label)
        # ===================================================
        images = batch['image'][le:].to(accelerator.device)
        output = model(images)
        new_dict = {task: output[task].detach().cpu().numpy() for task in p.ALL_TASKS.NAMES}
        label = np.ones(batch['image'].shape[0])
        attack_x.append(new_dict)
        attack_y.append(label)
        accelerator.print(f'{num} train data has labeled!')
        num += 1
    num = 0
    for batch in test:
        if num >= keep_num:
            break
        le = int(batch['image'].shape[0] / 2)
        images = batch['image'][0:le].to(accelerator.device)
        output = model(images)
        new_dict = {task: output[task].detach().cpu().numpy() for task in p.ALL_TASKS.NAMES}
        label = np.ones(batch['image'].shape[0])
        attack_x.append(new_dict)
        attack_y.append(label)
        # ===================================================
        images = batch['image'][le:].to(accelerator.device)
        output = model(images)
        new_dict = {task: output[task].detach().cpu().numpy() for task in p.ALL_TASKS.NAMES}
        label = np.ones(batch['image'].shape[0])
        attack_x.append(new_dict)
        attack_y.append(label)
        accelerator.print(f'{num} test data has labeled!')
        num += 1
    if shadow == True:
        save_dir = save_dir + '/shadow.pkl'
    else:
        save_dir = save_dir + '/target.pkl'
    save_data_to_local(data=attack_x, label=attack_y, file_path=save_dir)


def load_and_shuffle_data(file_path):
    def load_data_from_local(file_path):
        # 从文件中加载字节流
        with open(file_path, 'rb') as f:
            data_bytes = f.read()

        # 将字节流反序列化为元组
        data_tuple = pickle.loads(data_bytes)

        # 获取数据和标签
        data, label = data_tuple

        return data, label

    # 从本地加载数据和标签
    data, label = load_data_from_local(file_path)

    # 将列表中的数据和标签分别存入两个新列表
    data_list, label_list = [], []
    for d, l in zip(data, label):
        data_list.append(d)
        label_list.append(l)

    # 计算输入数据长度
    data_len = len(data_list)

    # 打乱数据排序
    idx = list(range(data_len))
    random.shuffle(idx)
    # 按照随机排序将数据和标签重新组合为新的列表
    data_list = [data_list[i] for i in idx]
    label_list = [label_list[i] for i in idx]

    return data_list, label_list


def print_label(model, dataset, accelerator, save_path='data.npz'):
    model.eval()
    train, test = dataset
    input_tensors = []
    label_tensors = []
    with torch.no_grad():
        for i in range(0, len(test)):
            batch = train[i]
            non_batch = test[i]
            # for i, batch in enumerate(ttrain):
            # Forward pass
            images = batch['image'].to(accelerator.device)
            targets = {task: batch[task].to(accelerator.device) for task in p.ALL_TASKS.NAMES}
            non_images = non_batch['image'].to(accelerator.device)
            non_targets = {task: non_batch[task].to(accelerator.device) for task in p.ALL_TASKS.NAMES}

            output = model(images)
            non_output = model(non_images)

            targets = torch.cat((targets['semseg'], targets['depth']), dim=1)
            non_targets = torch.cat((non_targets['semseg'], non_targets['depth']), dim=1)

            output = torch.cat((images, output['semseg'], output['depth'], targets), dim=1)
            non_output = torch.cat((non_images, non_output['semseg'], non_output['depth'], non_targets), dim=1)

            diff_inpput = torch.cat((output, non_output), dim=0)
            diff_target = torch.cat((torch.ones(output.size(0)), torch.zeros(non_output.size(0))), dim=0)

            # random
            rand_indices = torch.randperm(diff_inpput.size(0))
            diff_inpput = diff_inpput[rand_indices]
            diff_target = diff_target[rand_indices]

            # add list
            input_tensors.append(diff_inpput.to('cpu'))
            label_tensors.append(diff_target.to('cpu'))
            accelerator.print(f'{i + 1} data has been print!')

    # stack
    input_tensor = np.stack(input_tensors, axis=0)
    label_tensor = np.stack(label_tensors, axis=0)

    # save
    np.savez(save_path, input_tensor=input_tensor, label_tensor=label_tensor)


def load_label_data(save_path):
    data = np.load(save_path)
    loaded_input_tensor = data['input_tensor']
    loaded_label_tensor = data['label_tensor']
    return loaded_input_tensor, loaded_label_tensor


def save_csv(data, mood='data', img_path='.', dataset_choose='nyud'):
    def get_min_len(datas):
        small_data = []
        min_len = 1000000000
        for dats in datas:
            if len(dats)<min_len:
                min_len = len(dats)
                small_data = dats
        return small_data

    if mood == 'data':
        if dataset_choose == 'nyud':
            data1, data2, data3, data4 = data
            img_path = img_path + '/' + 'data.csv'
            with open(img_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Index', 'Non_Mem_Seg', 'Non_Mem_Dep', 'Mem_Seg', 'Mem_Dep'])  # 写入表头
                for i in range(len(data1)):
                    writer.writerow([i, data1[i], data2[i], data3[i], data4[i]])
        else:
            data1, data2, data3, data4, data5, data6, data7, data8 = data
            img_path = img_path + '/' + 'data.csv'
            with open(img_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    ['Index', 'Non_Mem_Seg', 'Non_Mem_Hp', 'Non_Mem_Sal', 'Non_Mem_Norm', 'Mem_Seg', 'Mem_Hp',
                     'Mem_Sal', 'Mem_Norm'])  # 写入表头
                count_data = get_min_len((data1, data2, data3, data4, data5, data6, data7, data8))
                for i in range(len(count_data)):
                    writer.writerow([i, data1[i], data2[i], data3[i], data4[i], data5[i], data6[i], data7[i], data8[i]])
    else:
        data1, data2 = data
        img_path = img_path + '/' + 'diff.csv'
        with open(img_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Mem_Diff', 'Non_Mem_Diff'])  # 写入表头
            count_data = get_min_len((data1, data2))
            for i in range(len(count_data)):
                writer.writerow([i, data1[i], data2[i]])  # 写入每行数据


def pring_img(data, mood='data', img_path='.', dataset_choose='nyud'):
    save_csv(data, mood, img_path, dataset_choose)
    if mood == 'data':
        # 读取数据，并绘制柱状图
        csv_path = img_path + '/' + 'data.csv'
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过表头
            indices = []
            data_values = []
            for row in reader:
                indices.append(int(row[0]))
                data_values.append([round(float(x), 2) for x in row[1:]])

        # 将数据转置，使得每一列对应一组数据
        data_values = np.array(data_values).T

        # 绘制柱状图
        bar_width = 0.3
        opacity = 0.8
        index = np.arange(len(indices))

        if dataset_choose == 'nyud':
            # Semage
            image_path = img_path + 'seg_data.png'
            plt.bar(index, data_values[0], bar_width, label='Non_Mem_Seg', color='#87CEFA', alpha=opacity)
            # plt.bar(index, data_values[1], bar_width, label='Non_Mem_Dep', color='#87CEFA', alpha=opacity)
            plt.bar(index, data_values[2], bar_width, bottom=data_values[0], label='Mem_Seg', color='#FFA07A',
                    alpha=opacity)
            # plt.bar(index, data_values[3], bar_width, bottom=-data_values[1], label='Mem_Dep', color='#FFA07A',
            #         alpha=opacity)

            plt.xlabel('Sample Index')
            plt.ylabel('MIOU Sorce Data')
            plt.title('Semage Sorce Numerical')
            plt.xticks(index, indices)
            plt.legend()  # 显示图例
            plt.axhline(0, color='black', linewidth=0.5)  # 添加分割线
            plt.tight_layout()
            plt.show()
            plt.savefig(image_path)

            # Dp
            image_path = img_path + 'dp_data.png'
            # plt.bar(index, data_values[0], bar_width, label='Non_Mem_Seg', color='#87CEFA', alpha=opacity)
            plt.bar(index, data_values[1], bar_width, label='Non_Mem_Dp', color='#87CEFA', alpha=opacity)
            # plt.bar(index, data_values[2], bar_width, bottom=data_values[0], label='Mem_Seg', color='#FFA07A',
            #         alpha=opacity)
            plt.bar(index, data_values[3], bar_width, bottom=-data_values[1], label='Mem_Dp', color='#FFA07A',
                    alpha=opacity)

            plt.xlabel('Sample Index')
            plt.ylabel('RMSE Sorce Data')
            plt.title('Depth Sorce Numerical')
            plt.xticks(index, indices)
            plt.legend()  # 显示图例
            plt.axhline(0, color='black', linewidth=0.5)  # 添加分割线
            plt.tight_layout()
            plt.show()
            plt.savefig(image_path)
        else:
            # Semage
            image_path = img_path + 'seg_data.png'
            plt.bar(index, data_values[0], bar_width, label='Non_Mem_Seg', color='#87CEFA', alpha=opacity)
            # plt.bar(index, data_values[1], bar_width, label='Non_Mem_Dep', color='#87CEFA', alpha=opacity)
            plt.bar(index, data_values[4], bar_width, bottom=data_values[0], label='Mem_Seg', color='#FFA07A',
                    alpha=opacity)
            # plt.bar(index, data_values[3], bar_width, bottom=-data_values[1], label='Mem_Dep', color='#FFA07A',
            #         alpha=opacity)

            plt.xlabel('Sample Index')
            plt.ylabel('MIOU Sorce Data')
            plt.title('Semage Sorce Numerical')
            plt.xticks(index, indices)
            plt.legend()  # 显示图例
            plt.axhline(0, color='black', linewidth=0.5)  # 添加分割线
            plt.tight_layout()
            plt.show()
            plt.savefig(image_path)

            # Hp
            image_path = img_path + 'hp_data.png'
            # plt.bar(index, data_values[0], bar_width, label='Non_Mem_Seg', color='#87CEFA', alpha=opacity)
            plt.bar(index, data_values[1], bar_width, label='Non_Mem_Hp', color='#87CEFA', alpha=opacity)
            # plt.bar(index, data_values[2], bar_width, bottom=data_values[0], label='Mem_Seg', color='#FFA07A',
            #         alpha=opacity)
            plt.bar(index, data_values[5], bar_width, bottom=-data_values[1], label='Mem_Hp', color='#FFA07A',
                    alpha=opacity)

            plt.xlabel('Sample Index')
            plt.ylabel('MIOU Sorce Data')
            plt.title('Human Part Sorce Numerical')
            plt.xticks(index, indices)
            plt.legend()  # 显示图例
            plt.axhline(0, color='black', linewidth=0.5)  # 添加分割线
            plt.tight_layout()
            plt.show()
            plt.savefig(image_path)

            # Sal
            image_path = img_path + 'sal_data.png'
            # plt.bar(index, data_values[0], bar_width, label='Non_Mem_Seg', color='#87CEFA', alpha=opacity)
            plt.bar(index, data_values[2], bar_width, label='Non_Mem_Sal', color='#87CEFA', alpha=opacity)
            # plt.bar(index, data_values[2], bar_width, bottom=data_values[0], label='Mem_Seg', color='#FFA07A',
            #         alpha=opacity)
            plt.bar(index, data_values[6], bar_width, bottom=-data_values[1], label='Mem_Sal', color='#FFA07A',
                    alpha=opacity)

            plt.xlabel('Sample Index')
            plt.ylabel('MIOU Sorce Data')
            plt.title('Sal Sorce Numerical')
            plt.xticks(index, indices)
            plt.legend()  # 显示图例
            plt.axhline(0, color='black', linewidth=0.5)  # 添加分割线
            plt.tight_layout()
            plt.show()
            plt.savefig(image_path)

            # Norm
            image_path = img_path + 'norm_data.png'
            # plt.bar(index, data_values[0], bar_width, label='Non_Mem_Seg', color='#87CEFA', alpha=opacity)
            plt.bar(index, data_values[3], bar_width, label='Non_Mem_Norm', color='#87CEFA', alpha=opacity)
            # plt.bar(index, data_values[2], bar_width, bottom=data_values[0], label='Mem_Seg', color='#FFA07A',
            #         alpha=opacity)
            plt.bar(index, data_values[7], bar_width, bottom=-data_values[1], label='Mem_Norm', color='#FFA07A',
                    alpha=opacity)

            plt.xlabel('Sample Index')
            plt.ylabel('mErr Sorce Data')
            plt.title('Norm Sorce Numerical')
            plt.xticks(index, indices)
            plt.legend()  # 显示图例
            plt.axhline(0, color='black', linewidth=0.5)  # 添加分割线
            plt.tight_layout()
            plt.show()
            plt.savefig(image_path)

    else:
        csv_path = img_path + '/' + 'diff.csv'
        img_path = img_path + 'diff.png'
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过表头
            indices = []
            data1_values = []
            data2_values = []
            for row in reader:
                indices.append(int(row[0]))
                data1_values.append(round(float(row[1]), 2))
                data2_values.append(round(float(row[2]), 2))

        # 绘制柱状图
        plt.bar(indices, data1_values, label='Mem_Differential', color='#FFA07A', zorder=0)
        plt.bar(indices, data2_values, label='Non_Mem_Differential', color='#87CEFA', zorder=1)
        plt.xlabel('Sample Index')
        plt.ylabel('Differential Data')
        plt.title('Differential Numerical')
        plt.savefig(img_path)
        plt.legend()  # 显示图例
        plt.show()


def give_data(input_tensor, dataset_choose='nyud'):
    if dataset_choose == 'nyud':
        orimage_tensor = input_tensor[:, :3, :, :]
        semseg_tensor = input_tensor[:, 3:43, :, :]
        depth_tensor = input_tensor[:, 43:44, :, :]
        target_tensor = input_tensor[:, 44:46, :, :]
        return orimage_tensor, semseg_tensor, depth_tensor, target_tensor
    else:
        orimage_tensor = input_tensor[:, :3, :, :]
        semseg_tensor = input_tensor[:, 3:24, :, :]
        hp_tensor = input_tensor[:, 24:31, :, :]
        sal_tensor = input_tensor[:, 31:32, :, :]
        norm_tensor = input_tensor[:, 32:35, :, :]
        target_tensor = input_tensor[:, 35:41, :, :]
        return orimage_tensor, semseg_tensor, hp_tensor, sal_tensor, norm_tensor, target_tensor


def give_threshold(pred, gt, task='semage'):
    def give_semage_test(pred, gt):
        segmeter = SemsegMeter(database='NYUD')
        pred = pred.permute(0, 2, 3, 1)
        _, pred = torch.max(pred, dim=3)
        segmeter.update(pred, gt)
        sorce = segmeter.get_score(verbose=False)
        acc = sorce['mIoU']
        return acc

    def give_depth_test(pred, gt):
        pred = pred.permute(0, 2, 3, 1)
        depmeter = DepthMeter()
        depmeter.update(pred, gt)
        sorce = depmeter.get_score(verbose=False)
        acc = sorce['rmse']
        return acc

    def give_human_test(pred, gt):
        pred = pred.permute(0, 2, 3, 1)
        _, pred = torch.max(pred, dim=3)
        n_parts = 6
        tp = [0] * (n_parts + 1)
        fp = [0] * (n_parts + 1)
        fn = [0] * (n_parts + 1)
        pred, gt = pred.squeeze(), gt.squeeze()
        valid = (gt != 255)

        for i_part in range(n_parts + 1):
            tmp_gt = (gt == i_part)
            tmp_pred = (pred == i_part)
            tp[i_part] += torch.sum(tmp_gt & tmp_pred & (valid)).item()
            fp[i_part] += torch.sum(~tmp_gt & tmp_pred & (valid)).item()
            fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & (valid)).item()

        jac = [0] * (n_parts + 1)
        for i_part in range(0, n_parts + 1):
            jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

        acc = np.mean(jac)

        return acc

    def give_sal_test(pred, gt):
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.squeeze(255 * 1 / (1 + torch.exp(-pred)))
        salmeter = SaliencyMeter()
        salmeter.update(pred, gt)
        acc = salmeter.get_score(verbose=False)['mIoU']
        return acc

    def give_norm_test(pred, gt):
        pred = pred.permute(0, 2, 3, 1)
        pred = (F.normalize(pred, p=2, dim=3) + 1.0) * 255 / 2.0
        pred = 2 * pred / 255 - 1
        pred = pred.permute(0, 3, 1, 2)  # [B, C, H, W]
        valid_mask = (gt != 255)
        invalid_mask = (gt == 255)

        # Put zeros where mask is invalid
        pred[invalid_mask] = 0.0
        gt[invalid_mask] = 0.0

        # Calculate difference expressed in degrees
        deg_diff_tmp = (180 / math.pi) * (torch.acos(torch.clamp(torch.sum(pred * gt, 1), min=-1, max=1)))
        deg_diff_tmp = torch.masked_select(deg_diff_tmp, valid_mask[:, 0])
        if deg_diff_tmp.numel() == 0:
            dul = 1
        else:
            dul = deg_diff_tmp.numel()
        acc = torch.sum(deg_diff_tmp).item()/dul
        # nometer = NormalsMeter()
        # nometer.update(pred, gt)
        # acc = nometer.get_score(verbose=False)['mean']
        return acc

    if task == 'semage':
        return give_semage_test(pred, gt)
    elif task == 'depth':
        return give_depth_test(pred, gt)
    elif task == 'human_part':
        return give_human_test(pred, gt)
    elif task == 'sal':
        return give_sal_test(pred, gt)
    else:
        return give_norm_test(pred, gt)


def give_task_list():
    seg_list = []
    dep_list = []
    hp_list = []
    sal_list = []
    norm_list = []
    return seg_list, dep_list, hp_list, sal_list, norm_list


def get_batch(model, train, test, idx):
    def give_task_cat(pard):
        def test_none(tes, cat_tes):
            if tes == None:
                return cat_tes
            else:
                return torch.cat((tes, cat_tes), dim=1)

        new_pard = None
        if 'semseg' in pard.keys():
            new_pard = test_none(new_pard, pard['semseg'])
        if 'depth' in pard.keys():
            new_pard = test_none(new_pard, pard['depth'])
        if 'human_parts' in pard.keys():
            new_pard = test_none(new_pard, pard['human_parts'])
        if 'sal' in pard.keys():
            new_pard = test_none(new_pard, pard['sal'])
        if 'normals' in pard.keys():
            new_pard = test_none(new_pard, pard['normals'])
        return new_pard

    model.eval()
    batch = train[idx]
    non_batch = test[idx]
    # Forward pass
    images = batch['image'].to(accelerator.device)
    targets = {task: batch[task].to(accelerator.device) for task in p.ALL_TASKS.NAMES}
    non_images = non_batch['image'].to(accelerator.device)
    non_targets = {task: non_batch[task].to(accelerator.device) for task in p.ALL_TASKS.NAMES}

    output = model(images)
    non_output = model(non_images)

    # targets = torch.cat((targets['semseg'], targets['depth']), dim=1)
    targets = give_task_cat(targets)
    # non_targets = torch.cat((non_targets['semseg'], non_targets['depth']), dim=1)
    non_targets = give_task_cat(non_targets)

    output = give_task_cat(output)
    output = torch.cat((images, output, targets), dim=1)
    non_output = give_task_cat(non_output)
    non_output = torch.cat((non_images, non_output, non_targets), dim=1)

    diff_input = torch.cat((output, non_output), dim=0)
    diff_target = torch.cat((torch.ones(output.size(0)), torch.zeros(non_output.size(0))), dim=0)

    # random
    rand_indices = torch.randperm(diff_input.size(0))
    diff_input = diff_input[rand_indices]
    diff_target = diff_target[rand_indices]

    return diff_input, diff_target


def get_attack_model_input(input_tensor, dataset_choose, alptho=3):
    def get_sum(input):
        return torch.unsqueeze(torch.sum(input, dim=1) / input.size(1), dim=1)

    if dataset_choose == 'nyud':
        orimage_tensor, semseg_tensor, depth_tensor, _ = give_data(input_tensor,
                                                                   dataset_choose)
        orimage_tensor = get_sum(orimage_tensor)
        semseg_tensor = get_sum(semseg_tensor)
        depth_tensor = get_sum(depth_tensor)
        input = orimage_tensor + alptho * torch.abs(semseg_tensor + depth_tensor)
    else:
        orimage_tensor, semseg_tensor, hp_tensor, sal_tensor, norm_tensor, _ = give_data(input_tensor,
                                                                                         dataset_choose)
        orimage_tensor = get_sum(orimage_tensor)
        semseg_tensor = get_sum(semseg_tensor)
        hp_tensor = get_sum(hp_tensor)
        sal_tensor = get_sum(sal_tensor)
        norm_tensor = get_sum(norm_tensor)
        input = orimage_tensor + alptho * torch.abs(semseg_tensor + hp_tensor + sal_tensor + norm_tensor)
    return input


def AMI(p, model, dataset, accelerator, attack_alptho=3):
    target_model, shadow_model = model
    dataset_choose = p['dataset']
    ((t_train, t_test), (s_train, s_test)) = dataset
    attack_model = get_resnet_classification_model(p, arch="resnet18", input_channel=1, num_classes=2,
                                                   dilated=True)
    optimizer = optim.Adam(attack_model.parameters(), lr=0.0001)
    attack_model, optimizer = accelerator.prepare(attack_model, optimizer)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    all_loss = 0
    for epoch in range(0, 30):
        # 训练
        attack_model.train()
        train_num = 0
        for i in range(0, len(s_train)):
            s_input, s_target = get_batch(shadow_model, s_train, s_test, i)
            s_input = s_input.type(torch.FloatTensor).to(accelerator.device)
            s_target = s_target.type(torch.LongTensor).to(accelerator.device)
            input = get_attack_model_input(s_input, dataset_choose, attack_alptho)
            optimizer.zero_grad()
            output = attack_model(input)
            loss = criterion(output, s_target)
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
            train_num += 1
        all_loss = all_loss / train_num
        # 验证
        test_num = 0
        test_acc = 0
        attack_model.eval()
        with torch.no_grad():
            for i in range(0, len(t_train)):
                t_input, t_target = get_batch(target_model, t_train, t_test, i)
                t_input = t_input.type(torch.FloatTensor).to(accelerator.device)
                t_target = t_target.type(torch.LongTensor).to(accelerator.device)
                input = get_attack_model_input(t_input, dataset_choose, attack_alptho)
                output = attack_model(input).argmax(dim=1)
                test_acc += torch.eq(output, t_target).float().sum().item() / output.size(0)
                test_num += 1
            test_acc = test_acc / test_num
        # test_num = 0
        train_acc = 0
        # with torch.no_grad():
        #     for i in range(0, len(s_train)):
        #         s_input, s_target = get_batch(shadow_model, s_train, s_test, i)
        #         s_input = s_input.type(torch.FloatTensor).to(accelerator.device)
        #         s_target = s_target.type(torch.LongTensor).to(accelerator.device)
        #         input = get_attack_model_input(s_input, dataset_choose, attack_alptho)
        #         output = attack_model(input).argmax(dim=1)
        #         train_acc += torch.eq(output, s_target).float().sum().item() / output.size(0)
        #         test_num += 1 * s_input.size(0)
        #     train_acc = train_acc / test_num
        # 计算最优精度
        if best_acc < test_acc:
            best_acc = test_acc

        accelerator.print(
            f'[{epoch + 1}/{30}] Best attack acc: {best_acc * 100} %  Now Train acc: {train_acc * 100}% Test acc: {test_acc * 100}% Loss:{all_loss}')
    accelerator.print(f'Attack acc: {best_acc * 100} %')
    return best_acc


def TI(p, model, dataset, accelerator, img_path):
    dataset_choose = p['dataset']
    ((t_train, t_test), (s_train, s_test)) = dataset
    target_model, shadow_model = model
    tr_seg_list, tr_dep_list, tr_hp_list, tr_sal_list, tr_norm_list = give_task_list()
    te_seg_list, te_dep_list, te_hp_list, te_sal_list, te_norm_list = give_task_list()
    tr_mu_score = []
    te_mu_score = []

    # 获得threshold
    with torch.no_grad():
        for i in range(0, len(s_train)):
            s_input, s_target = get_batch(shadow_model, s_train, s_test, i)
            s_input = s_input.type(torch.FloatTensor).to(accelerator.device)
            s_target = s_target.type(torch.LongTensor).to(accelerator.device)
            if dataset_choose == 'nyud':
                tr_orimage_tensor, tr_semseg_tensor, tr_depth_tensor, tr_target_tensor = give_data(s_input,
                                                                                                   dataset_choose)
                for j in range(0, s_target.size(0)):
                    s = torch.unsqueeze(tr_semseg_tensor[j], dim=0)
                    s_t = torch.unsqueeze(tr_target_tensor[:, 0:1, :, :][j], dim=0)
                    d = torch.unsqueeze(tr_depth_tensor[j], dim=0)
                    d_t = torch.unsqueeze(tr_target_tensor[:, 1:2, :, :][j], dim=0)
                    # diff_guess = torch.mean(torch.abs(torch.abs(semseg_tensor) - depth_tensor))
                    seg_acc = give_threshold(s, s_t, task='semage')
                    dep_acc = give_threshold(d, d_t, task='depth')
                    result = {'semseg': {'mIoU': seg_acc.item()}, 'depth': {'rmse': dep_acc.item()}}
                    mu_score = calculate_multi_task_performance(result)
                    membership_label = s_target[j]

                    if membership_label == 1:
                        accelerator.print(
                            f'membership accs: seg: {seg_acc.item()}, de:{dep_acc.item()}, score:{mu_score}')
                        tr_seg_list.append(seg_acc.item())
                        tr_dep_list.append(dep_acc.item())
                        tr_mu_score.append(mu_score)
                    else:
                        accelerator.print(
                            f'non membership accs: seg: {seg_acc.item()}, de:{dep_acc.item()}, score:{mu_score}')
                        te_seg_list.append(seg_acc.item())
                        te_dep_list.append(dep_acc.item())
                        te_mu_score.append(mu_score)
            else:
                tr_orimage_tensor, tr_semseg_tensor, tr_hp_tensor, tr_sal_tensor, tr_norm_tensor, tr_target_tensor = give_data(
                    s_input, dataset_choose)

                for j in range(0, s_target.size(0)):
                    s = torch.unsqueeze(tr_semseg_tensor[j], dim=0)
                    s_t = torch.unsqueeze(tr_target_tensor[:, 0:1, :, :][j], dim=0)
                    hp = torch.unsqueeze(tr_hp_tensor[j], dim=0)
                    hp_t = torch.unsqueeze(tr_target_tensor[:, 1:2, :, :][j], dim=0)
                    sal = torch.unsqueeze(tr_sal_tensor[j], dim=0)
                    sal_t = torch.unsqueeze(tr_target_tensor[:, 2:3, :, :][j], dim=0)
                    norm = torch.unsqueeze(tr_norm_tensor[j], dim=0)
                    norm_t = torch.unsqueeze(tr_target_tensor[:, 3:6, :, :][j], dim=0)
                    # diff_guess = torch.mean(torch.abs(torch.abs(semseg_tensor) - depth_tensor))
                    seg_acc = give_threshold(s, s_t, task='semage')
                    hp_acc = give_threshold(hp, hp_t, task='human_part')
                    sal_acc = give_threshold(sal, sal_t, task='sal')
                    norm_acc = give_threshold(norm, norm_t, task='norm')
                    result = {'semseg': {'mIoU': seg_acc.item()},
                              'human_parts': {'mIoU': hp_acc.item()},
                              'sal': {'mIoU': sal_acc},
                              'normals': {'mean': norm_acc}}
                    mu_score = calculate_multi_task_performance(result)
                    membership_label = s_target[j]
                    if membership_label == 1:
                        tr_seg_list.append(seg_acc.item())
                        tr_sal_list.append(sal_acc)
                        if hp_acc.item() != 0 and norm_acc != 0:
                            accelerator.print(
                                f'membership accs: score:{mu_score}, seg: {seg_acc.item()}, hp: {hp_acc.item()}, sal: {sal_acc}, norm: {norm_acc}')
                            tr_mu_score.append(mu_score)
                            tr_norm_list.append(norm_acc)
                            tr_hp_list.append(hp_acc.item())
                        else:
                            accelerator.print(
                                f'membership accs: score:{0}, seg: {seg_acc.item()}, hp: {hp_acc.item()}, sal: {sal_acc}, norm: {norm_acc}')

                    else:
                        te_seg_list.append(seg_acc.item())
                        te_sal_list.append(sal_acc)
                        if hp_acc.item() != 0 and norm_acc != 0:
                            accelerator.print(
                                f'non membership accs: score:{mu_score}, seg: {seg_acc.item()}, hp: {hp_acc.item()}, sal: {sal_acc}, norm: {norm_acc}')
                            te_mu_score.append(mu_score)
                            te_norm_list.append(norm_acc)
                            te_hp_list.append(hp_acc.item())
                        else:
                            accelerator.print(
                                f'non membership accs: score:{0}, seg: {seg_acc.item()}, hp: {hp_acc.item()}, sal: {sal_acc}, norm: {norm_acc}')

    if dataset_choose == 'nyud':
        membership_seg_threshold = max(te_seg_list)
        membership_dep_threshold = min(te_dep_list)
        # pring_img(data=(te_seg_list, te_dep_list, tr_seg_list, tr_dep_list), mood='data', img_path=img_path,
        #           dataset_choose=dataset_choose)
    else:
        membership_seg_threshold = max(te_seg_list)
        membership_hp_threshold = max(te_hp_list)
        membership_sal_threshold = max(te_sal_list)
        membership_norm_threshold = min(te_norm_list)
        # pring_img(data=(
        #     te_seg_list, te_hp_list, te_sal_list, te_norm_list, tr_seg_list, tr_hp_list, tr_sal_list,
        #     tr_norm_list),
        #     mood='data', img_path=img_path, dataset_choose=dataset_choose)
    membership_score_threshold = max(te_mu_score)
    # pring_img(data=(tr_mu_score, te_mu_score), mood='score', img_path=img_path, dataset_choose=dataset_choose)
    with torch.no_grad():
        attack_acc = 0
        attack_num = 0
        for i in range(0, len(t_train)):
            t_input, t_target = get_batch(target_model, t_train, t_test, i)
            t_input = t_input.type(torch.FloatTensor).to(accelerator.device)
            t_target = t_target.type(torch.LongTensor).to(accelerator.device)
            guess_label_tensor = torch.zeros(t_target.size())
            if dataset_choose == 'nyud':
                te_orimage_tensor, te_semseg_tensor, te_depth_tensor, te_target_tensor = give_data(t_input,
                                                                                                   dataset_choose)
                for j in range(0, t_target.size(0)):
                    s = torch.unsqueeze(te_semseg_tensor[j], dim=0)
                    s_t = torch.unsqueeze(te_target_tensor[:, 0:1, :, :][j], dim=0)
                    d = torch.unsqueeze(te_depth_tensor[j], dim=0)
                    d_t = torch.unsqueeze(te_target_tensor[:, 1:2, :, :][j], dim=0)
                    seg_acc = give_threshold(s, s_t, task='semage')
                    dep_acc = give_threshold(d, d_t, task='depth')
                    # diff_guess = torch.mean(torch.abs(torch.abs(semseg_tensor) - depth_tensor))
                    if seg_acc.item() > membership_seg_threshold or dep_acc.item() < membership_dep_threshold:
                        guess_label_tensor[j] = 1.
                    result = {'semseg': {'mIoU': seg_acc.item()}, 'depth': {'rmse': dep_acc.item()}}
                    mu_score = calculate_multi_task_performance(result)
                    if mu_score > membership_score_threshold:
                        guess_label_tensor[j] = 1.
            else:
                te_orimage_tensor, te_semseg_tensor, te_hp_tensor, te_sal_tensor, te_norm_tensor, te_target_tensor = give_data(
                    t_input, dataset_choose)

                for j in range(0, s_target.size(0)):
                    s = torch.unsqueeze(te_semseg_tensor[j], dim=0)
                    s_t = torch.unsqueeze(te_target_tensor[:, 0:1, :, :][j], dim=0)
                    hp = torch.unsqueeze(te_hp_tensor[j], dim=0)
                    hp_t = torch.unsqueeze(te_target_tensor[:, 1:2, :, :][j], dim=0)
                    sal = torch.unsqueeze(te_sal_tensor[j], dim=0)
                    sal_t = torch.unsqueeze(te_target_tensor[:, 2:3, :, :][j], dim=0)
                    norm = torch.unsqueeze(te_norm_tensor[j], dim=0)
                    norm_t = torch.unsqueeze(te_target_tensor[:, 3:6, :, :][j], dim=0)
                    # diff_guess = torch.mean(torch.abs(torch.abs(semseg_tensor) - depth_tensor))
                    seg_acc = give_threshold(s, s_t, task='semage')
                    hp_acc = give_threshold(hp, hp_t, task='human_part')
                    sal_acc = give_threshold(sal, sal_t, task='sal')
                    norm_acc = give_threshold(norm, norm_t, task='norm')
                    result = {'semseg': {'mIoU': seg_acc.item()},
                              'human_parts': {'mIoU': hp_acc.item()},
                              'sal': {'mIoU': sal_acc},
                              'normals': {'mean': norm_acc}}
                    mu_score = calculate_multi_task_performance(result)
                    if hp_acc.item() != 0 and norm_acc != 0 and mu_score > membership_score_threshold:
                        guess_label_tensor[j] = 1.
                    elif hp_acc.item() != 0 and hp_acc.item() > membership_hp_threshold:
                        guess_label_tensor[j] = 1.
                    elif norm_acc != 0.0 and norm_acc < membership_norm_threshold:
                        guess_label_tensor[j] = 1.
                    elif seg_acc.item() > membership_seg_threshold or sal_acc > membership_sal_threshold:
                        guess_label_tensor[j] = 1.

            print('guess_label_tensor: ', guess_label_tensor)
            print('target_label_tensor: ', t_target)
            acc = torch.eq(guess_label_tensor.to('cpu'),
                           t_target.to('cpu')).float().sum().item() / guess_label_tensor.size(0)
            attack_acc += acc
            print(f'attack acc: {acc}')
            attack_num += 1
        attack_acc = attack_acc / attack_num
    return attack_acc





if __name__ == '__main__':
    logging_dir = os.getcwd() + '/logs/' + str(datetime.datetime.now())  # 生成logs记录文件文件夹
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"],project_dir=logging_dir)  # 多卡训练框架
    Logger(logging_dir if accelerator.is_local_main_process else None)  # 用以产生训练记录文件
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(args))  # 打印参数信息

    # 提供训练指导
    config_exp = give_config_exp(args.trainer.dataset, args.trainer.model_choose)

    p = create_config(args.trainer.defence_dir + '/target_results', config_exp)
    sub_p = create_config(args.trainer.defence_dir + '/shadow_results', config_exp)

    p['dataset'] = args.trainer.dataset
    sub_p['dataset'] = args.trainer.dataset
    p['debug'] = args.trainer.debug
    sub_p['debug'] = args.trainer.debug
    p['10_eval'] = args.trainer.ten_eval
    sub_p['10_eval'] = args.trainer.ten_eval

    accelerator.print(colored(p, 'red'))
    # Get model
    accelerator.print(colored('Retrieve model', 'blue'))
    target_model = get_model(p)
    shadow_model = get_model(sub_p)

    # Get criterion
    accelerator.print(colored('Get loss', 'blue'))
    criterion = get_criterion(p)
    criterion.cuda()
    accelerator.print(criterion)

    # CUDNN
    accelerator.print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Optimizer
    accelerator.print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, target_model)
    optimizer_s = get_optimizer(sub_p, shadow_model)
    accelerator.print(optimizer)

    # Transforms
    train_transforms, val_transforms = get_transformations(p)

    # Dataset
    num_workers = args.trainer.num_workers
    batch_size = args.trainer.batch_size
    accelerator.print(colored('Retrieve dataset', 'blue'))
    # train_dataset = get_train_dataset(p, train_transforms, download=False)
    # val_dataset = get_val_dataset(p, val_transforms, download=False)
    # save_name = 'MIA_{}'.format(p['dataset'])
    # save_path = './datasets/MIAdata/'
    save_name = 'MIA_{}'.format(p['dataset'])
    save_path = './datasets/MIAdata/'
    t_train, t_test, s_train, s_test, t_size, s_size = division(dataset=p['dataset'], save_name=save_name,
                                                                save_path=save_path)

    accelerator.print('Train samples %d - Val samples %d' % (len(t_train), len(t_test)))
    accelerator.print('Train transformations:')
    accelerator.print(train_transforms)
    accelerator.print('Val transformations:')
    accelerator.print(val_transforms)

    # 加载模型与数据
    target_model, shadow_model, optimizer, optimizer_s = accelerator.prepare(target_model, shadow_model, optimizer,
                                                                             optimizer_s)

    p['resum'] = args.trainer.resum
    sub_p['resum'] = args.trainer.resum
    target_model, start_epoch, best_result = resum(p, target_model, t_test, accelerator)
    shadow_model, s_start_epoch, s_best_result = resum(sub_p, shadow_model, s_test, accelerator)
    # _, _ = eval_acc(p, target_model, (t_train, t_test), accelerator, shadow=False)
    if_once_epoch = 5
    similar_altho = 2
    segmentation_altho = 0.6
    target_model = defence_train_model(p=p, model=target_model, criterion=criterion, optimizer=optimizer,
                                       data=(t_train, t_test), start_epoch=start_epoch, best_result=best_result,
                                       accelerator=accelerator, similar_altho=similar_altho,if_once_epoch=if_once_epoch,
                                       segmentation_altho=segmentation_altho, shadow=False)
    shadow_model = defence_train_model(p=sub_p, model=shadow_model, criterion=criterion, optimizer=optimizer_s,
                                       data=(s_train, s_test), start_epoch=s_start_epoch, best_result=s_best_result,
                                       accelerator=accelerator, similar_altho=similar_altho,if_once_epoch=if_once_epoch,
                                       segmentation_altho=segmentation_altho, shadow=True)

    #  查看性能
    _, _ = eval_acc(p, target_model, (t_train, t_test), accelerator, shadow=False)
    _, _ = eval_acc(sub_p, shadow_model, (s_train, s_test), accelerator, shadow=True)

    model = (target_model, shadow_model)
    dataset = ((t_train, t_test), (s_train, s_test))
    save_name = 'MIA_{}_defence'.format(p['dataset'])
    save_path = './datasets/MIAdata/{}_defence/'.format(args.trainer.model_choose)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path + save_name
    img_path = './files/data_details/{}_{}_defence/'.format(p['dataset'], args.trainer.model_choose)

    ami_acc = AMI(p, model, dataset, accelerator, attack_alptho=1)
    ti_acc = TI(p, model, dataset, accelerator, img_path)
    _, _ = eval_acc(p, target_model, (t_train, t_test), accelerator, shadow=False)

    accelerator.print(f'AMI-MTL acc: {ami_acc * 100} %')
    accelerator.print(f'TI-MTL acc: {ti_acc * 100} %')
