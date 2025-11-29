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


def give_threshold(pred, gt, task='semseg'):
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
        acc = torch.tensor(acc)
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
        acc = torch.sum(deg_diff_tmp).item() / dul
        # nometer = NormalsMeter()
        # nometer.update(pred, gt)
        # acc = nometer.get_score(verbose=False)['mean']
        return acc

    if task == 'semseg':
        return give_semage_test(pred, gt)
    elif task == 'depth':
        return give_depth_test(pred, gt)
    elif task == 'human_part':
        return give_human_test(pred, gt)
    elif task == 'sal':
        return give_sal_test(pred, gt)
    else:
        return give_norm_test(pred, gt)


def give_target(pred, task='semseg'):
    if task == 'semseg':
        pred = pred.permute(0, 2, 3, 1)
        _, pred = torch.max(pred, dim=3)
        return pred.unsqueeze(1)
    elif task == 'depth':
        return pred


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
        # guide_task = model(images)
        # for key in guide_task.keys():
        #     if key == 'deep_supervision':
        #         continue
        #     guide_task[key] = guide_task[key].detach()
        # mem_task = model(images)

        # return test_acc(p, mem_task, guide_task)

    task_batch, non_task_batch = batch_segmentation(batch, segmentation_altho)
    task_loss = tasks_performance(p, model, task_batch, losses, criterion, performance_meter)
    similar_loss = 0
    if if_once_epoch == True:
        all_loss = task_loss
    else:
        similar_loss = similar_calculations(p, model, non_task_batch, task_batch, losses, criterion,
                                            performance_meter)
        all_loss = task_loss + similar_altho * similar_loss
    return all_loss,task_loss,similar_altho *similar_loss


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


def save_csv(data, mood='data', img_path='.'):
    if mood == 'data':
        data1, data2, data3, data4 = data
        img_path = img_path + '/' + 'data.csv'
        with open(img_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Non_Mem_Seg', 'Non_Mem_Dep', 'Mem_Seg', 'Mem_Dep'])  # 写入表头
            for i in range(len(data1)):
                writer.writerow([i, data1[i], data2[i], data3[i], data4[i]])
    else:
        data1, data2 = data
        img_path = img_path + '/' + 'diff.csv'
        with open(img_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Non_Mem_Diff', 'Mem_Diff'])  # 写入表头
            for i in range(len(data1)):
                writer.writerow([i, data1[i], data2[i]])  # 写入每行数据


def pring_img(data, mood='data', img_path='.'):
    save_csv(data, mood, img_path)
    if mood == 'data':
        # 读取数据，并绘制柱状图
        csv_path = img_path + '/' + 'data.csv'
        img_path = img_path + 'data.png'
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

        plt.bar(index, data_values[0], bar_width, label='Non_Mem_Seg', color='#87CEFA', alpha=opacity)
        plt.bar(index, data_values[1], bar_width, label='Non_Mem_Dep', color='#87CEFA', alpha=opacity)
        plt.bar(index, data_values[2], bar_width, bottom=data_values[0], label='Mem_Seg', color='#FFA07A',
                alpha=opacity)
        plt.bar(index, data_values[3], bar_width, bottom=-data_values[1], label='Mem_Dep', color='#FFA07A',
                alpha=opacity)

        plt.xlabel('Sample Index')
        plt.ylabel('Sorce Data')
        plt.title('Sorce Numerical')
        plt.xticks(index, indices)
        plt.legend()  # 显示图例
        plt.axhline(0, color='black', linewidth=0.5)  # 添加分割线
        plt.tight_layout()
        plt.show()
        plt.savefig(img_path)
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
        plt.bar(indices, data2_values, label='Mem_Differential', color='#FFA07A', zorder=0)
        plt.bar(indices, data1_values, label='Non_Mem_Differential', color='#87CEFA', zorder=1)
        plt.xlabel('Sample Index')
        plt.ylabel('Differential Data')
        plt.title('Differential Numerical')
        plt.legend()  # 显示图例
        plt.show()
        plt.savefig(img_path)


def give_data(input_tensor):
    orimage_tensor = input_tensor[:, :3, :, :]
    semseg_tensor = input_tensor[:, 3:43, :, :]
    depth_tensor = input_tensor[:, 43:44, :, :]
    target_tensor = input_tensor[:, 44:46, :, :]
    return orimage_tensor, semseg_tensor, depth_tensor, target_tensor


def train_attack_model(p, data, accelerator):
    attack_model = get_resnet_classification_model(p, arch="resnet18", input_channel=1, num_classes=2,
                                                   dilated=True)
    (train_input_tensor, train_label_tensor), (test_input_tensor, test_label_tensor) = data
    optimizer = optim.Adam(attack_model.parameters(), lr=0.0001)
    attack_model, optimizer = accelerator.prepare(attack_model, optimizer)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(0, 30):
        attack_model.train()
        all_loss = 0
        attack_model.train()
        for i in range(0, train_input_tensor.size(0)):
            tr_input_tensor = train_input_tensor[i].type(torch.FloatTensor).to(accelerator.device)
            tr_label_tensor = train_label_tensor[i].type(torch.LongTensor).to(accelerator.device)
            tr_orimage_tensor, tr_semseg_tensor, tr_depth_tensor, tr_target_tensor = give_data(tr_input_tensor)
            orimage_tensor = torch.unsqueeze(torch.sum(tr_orimage_tensor, dim=1) / tr_orimage_tensor.size(1), dim=1)
            semseg_tensor = torch.unsqueeze(torch.sum(tr_semseg_tensor, dim=1) / tr_semseg_tensor.size(1), dim=1)
            depth_tensor = torch.unsqueeze(torch.sum(tr_depth_tensor, dim=1) / tr_depth_tensor.size(1), dim=1)
            input = orimage_tensor + 3 * torch.abs(semseg_tensor + depth_tensor)
            # input = torch.cat((orimage_tensor,semseg_tensor,depth_tensor),dim=1)
            optimizer.zero_grad()
            output = attack_model(input)
            loss = criterion(output, tr_label_tensor)
            # loss_dict['total'].backward()
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
        all_loss = all_loss / train_input_tensor.size(0)
        train_acc = test_attack_model(attack_model, (train_input_tensor, train_label_tensor), accelerator)
        test_acc = test_attack_model(attack_model, (test_input_tensor, test_label_tensor), accelerator)
        if best_acc < test_acc:
            best_acc = test_acc
        accelerator.print(
            f'[{epoch + 1}/{30}] Best attack acc: {best_acc * 100} %  Now Train acc: {train_acc * 100}% Test acc: {test_acc * 100}% Loss:{all_loss}')
    accelerator.print(f'Attack acc: {best_acc * 100} %')
    return best_acc


def test_attack_model(attack_model, data, accelerator):
    (test_input_tensor, test_label_tensor) = data
    attack_model.eval()
    acc = 0
    for i in range(0, test_input_tensor.size(0)):
        tr_input_tensor = test_input_tensor[i].type(torch.FloatTensor).to(accelerator.device)
        tr_label_tensor = test_label_tensor[i].type(torch.LongTensor).to(accelerator.device)
        tr_orimage_tensor, tr_semseg_tensor, tr_depth_tensor, tr_target_tensor = give_data(tr_input_tensor)
        orimage_tensor = torch.unsqueeze(torch.sum(tr_orimage_tensor, dim=1) / tr_orimage_tensor.size(1), dim=1)
        semseg_tensor = torch.unsqueeze(torch.sum(tr_semseg_tensor, dim=1) / tr_semseg_tensor.size(1), dim=1)
        depth_tensor = torch.unsqueeze(torch.sum(tr_depth_tensor, dim=1) / tr_depth_tensor.size(1), dim=1)
        input = orimage_tensor + 3 * torch.abs(semseg_tensor + depth_tensor)
        # input = torch.cat((orimage_tensor, semseg_tensor, depth_tensor), dim=1)
        output = attack_model(input).argmax(dim=1)
        acc += torch.eq(output, tr_label_tensor).float().sum().item() / output.size(0)
    return acc / test_input_tensor.size(0)


def guess(datasets, accelerator, dataset='nyud', img_path='.csv'):
    def give_semage_test(pred, gt):
        segmeter = SemsegMeter(database='NYUD')
        pred = pred.permute(0, 2, 3, 1)
        _, pred = torch.max(pred, dim=3)
        segmeter.update(pred, gt)
        sorce = segmeter.get_score(verbose=False)
        acc = sorce['mIoU']
        return acc

    def give_depth_test(pred, gt):
        depmeter = DepthMeter()
        depmeter.update(pred, gt)
        sorce = depmeter.get_score(verbose=False)
        acc = sorce['rmse']
        return acc

    def give_diff_test(seg, dep):
        semseg_tensor = torch.unsqueeze(torch.sum(seg, dim=1) / seg.size(1), dim=1)
        depth_tensor = torch.unsqueeze(torch.sum(dep, dim=1) / dep.size(1), dim=1)

        diff_guess = torch.mean(torch.abs(semseg_tensor) - depth_tensor)
        return diff_guess

    shadow_data, target_data = datasets
    train_input_tensor, train_label_tensor = shadow_data
    test_input_tensor, test_label_tensor = target_data

    if dataset == 'nyud':
        tr_diff_list = []
        te_diff_list = []

        tr_seg_list = []
        tr_dep_list = []
        te_seg_list = []
        te_dep_list = []

        # 判断分割阈值
        with torch.no_grad():
            max_membership_seg_acc = 0
            max_non_membership_seg_acc = 0
            min_membership_dep_acc = 100
            min_non_membership_dep_acc = 100
            tr_min_diff_guess = 100
            te_min_diff_guess = 100
            tr_max_diff_guess = 0
            te_max_diff_guess = 0
            avg_threshold = 0
            test_sample = 0
            for i in range(0, train_input_tensor.size(0)):
                tr_input_tensor = train_input_tensor[i].type(torch.FloatTensor).to(accelerator.device)
                tr_label_tensor = train_label_tensor[i].type(torch.LongTensor).to(accelerator.device)
                tr_orimage_tensor, tr_semseg_tensor, tr_depth_tensor, tr_target_tensor = give_data(tr_input_tensor)

                for j in range(0, tr_label_tensor.size(0)):
                    tst = torch.unsqueeze(tr_semseg_tensor[j], dim=1)
                    tstt = torch.unsqueeze(tr_target_tensor[:, 0:1, :, :][j], dim=1)
                    trt = torch.unsqueeze(tr_depth_tensor[j], dim=1)
                    trtt = torch.unsqueeze(tr_target_tensor[:, 1:2, :, :][j], dim=1)

                    diff_guess = give_diff_test(tst, trt)
                    # diff_guess = torch.mean(torch.abs(torch.abs(semseg_tensor) - depth_tensor))
                    seg_acc = give_semage_test(tst, tstt)
                    dep_acc = give_depth_test(trt, trtt)
                    membership_label = tr_label_tensor[j]
                    test_sample += 1
                    if membership_label == 1:
                        tr_diff_list.append(diff_guess.item())
                        tr_seg_list.append(seg_acc.item())
                        tr_dep_list.append(dep_acc.item())
                        if diff_guess < tr_min_diff_guess:
                            tr_min_diff_guess = diff_guess
                        if diff_guess > tr_max_diff_guess:
                            tr_max_diff_guess = diff_guess
                        if max_membership_seg_acc < seg_acc:
                            max_membership_seg_acc = seg_acc
                        if min_membership_dep_acc > dep_acc:
                            min_membership_dep_acc = dep_acc
                    else:
                        te_diff_list.append(diff_guess.item())
                        te_seg_list.append(seg_acc.item())
                        te_dep_list.append(dep_acc.item())
                        avg_threshold += diff_guess
                        if diff_guess < te_min_diff_guess:
                            te_min_diff_guess = diff_guess
                        if diff_guess > te_max_diff_guess:
                            te_max_diff_guess = diff_guess
                        if max_non_membership_seg_acc < seg_acc:
                            max_non_membership_seg_acc = seg_acc
                        if min_non_membership_dep_acc > dep_acc:
                            min_non_membership_dep_acc = dep_acc

        avg_threshold = avg_threshold / test_sample
        membership_seg_threshold = max_non_membership_seg_acc
        membership_dep_threshold = min_non_membership_dep_acc

        attack_acc1 = 0
        attack_acc2 = 0

        with torch.no_grad():
            for i in range(0, test_input_tensor.size(0)):
                te_input_tensor = test_input_tensor[i].type(torch.FloatTensor).to(accelerator.device)
                te_label_tensor = test_label_tensor[i].type(torch.LongTensor).to(accelerator.device)
                te_orimage_tensor, te_semseg_tensor, te_depth_tensor, te_target_tensor = give_data(te_input_tensor)

                guess_label_tensor1 = torch.zeros(te_label_tensor.size())
                guess_label_tensor2 = torch.zeros(te_label_tensor.size())
                for j in range(0, tr_label_tensor.size(0)):
                    tst = torch.unsqueeze(te_semseg_tensor[j], dim=1)
                    tstt = torch.unsqueeze(te_target_tensor[:, 0:1, :, :][j], dim=1)
                    trt = torch.unsqueeze(te_depth_tensor[j], dim=1)
                    trtt = torch.unsqueeze(te_target_tensor[:, 1:2, :, :][j], dim=1)
                    diff_guess = give_diff_test(tst, trt)
                    te_seg_acc = give_semage_test(tst, tstt)
                    te_dep_acc = give_depth_test(trt, trtt)

                    # diff_guess = torch.mean(torch.abs(torch.abs(semseg_tensor) - depth_tensor))
                    if te_seg_acc > membership_seg_threshold or te_dep_acc < membership_dep_threshold:
                        guess_label_tensor1[j] = 1.

                    if (diff_guess > avg_threshold and diff_guess > te_max_diff_guess):
                        guess_label_tensor2[j] = 1.
                    elif (diff_guess < avg_threshold):
                        if te_min_diff_guess < 0 and tr_min_diff_guess < 0:
                            if torch.abs(diff_guess) > torch.abs(te_min_diff_guess):
                                guess_label_tensor2[j] = 1.
                        elif te_min_diff_guess > 0:
                            if diff_guess < te_min_diff_guess:
                                guess_label_tensor2[j] = 1.

                accelerator.print(f"[{i}/{test_input_tensor.size(0)}] Attack tensor1: {guess_label_tensor1.to('cpu')}]")
                accelerator.print(f"[{i}/{test_input_tensor.size(0)}] Attack tensor1: {guess_label_tensor2.to('cpu')}]")
                accelerator.print(f"[{i}/{test_input_tensor.size(0)}] Real tensor: {te_label_tensor.to('cpu')}] \n")
                attack_acc1 += torch.eq(guess_label_tensor1.to('cpu'),
                                        te_label_tensor.to('cpu')).float().sum().item() / guess_label_tensor1.size(0)
                attack_acc2 += torch.eq(guess_label_tensor2.to('cpu'),
                                        te_label_tensor.to('cpu')).float().sum().item() / guess_label_tensor1.size(0)
        attack_acc1 = attack_acc1 / test_input_tensor.size(0)
        attack_acc2 = attack_acc2 / test_input_tensor.size(0)
        pring_img(data=(te_seg_list, te_dep_list, tr_seg_list, tr_dep_list), mood='data', img_path=img_path)
        # pring_img(data=(tr_diff_list, te_diff_list), mood='diff', img_path=img_path)
        return attack_acc1, attack_acc2


def guess_membership_data(p, model, dataset, accelerator, save_path, img_path, dataset_choose='nyud',
                          print_again=False):
    p, sub_p = p
    target_model, shadow_model = model
    target_model, shadow_model = accelerator.prepare(target_model, shadow_model)
    (t_train, t_test), (s_train, s_test) = dataset
    if print_again == True or not (
            os.path.exists(save_path + '_defence_shadow.npz') or os.path.exists(save_path + '_defence_target.npz')):
        # shadow
        print_label(shadow_model, (s_train, s_test), accelerator, save_path + '_defence_shadow.npz')
        # target
        print_label(target_model, (t_train, t_test), accelerator, save_path + '_defence_target.npz')

    # load data
    # shadow
    shadow_data = load_label_data(save_path + '_defence_shadow.npz')
    # target
    target_data = load_label_data(save_path + '_defence_target.npz')

    # change to tensor
    train_input_tensor, train_label_tensor = shadow_data
    train_input_tensor, train_label_tensor = torch.from_numpy(train_input_tensor), torch.from_numpy(train_label_tensor)
    test_input_tensor, test_label_tensor = target_data
    test_input_tensor, test_label_tensor = torch.from_numpy(test_input_tensor), torch.from_numpy(test_label_tensor)
    shadow_data = (train_input_tensor, train_label_tensor)
    target_data = (test_input_tensor, test_label_tensor)

    # train model
    best_attack_model_acc = train_attack_model(p, (shadow_data, target_data), accelerator)
    # guess
    best_label_acc, best_diff_acc = guess((shadow_data, target_data), dataset=dataset_choose, img_path=img_path,
                                          accelerator=accelerator)
    # 打印
    accelerator.print(f'best_AMI_acc: {best_attack_model_acc * 100} %')
    accelerator.print(f'best_TI_acc: {best_label_acc * 100} %')


if __name__ == '__main__':
    logging_dir = os.getcwd() + '/logs/' + str(datetime.datetime.now())  # 生成logs记录文件文件夹
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], project_dir=logging_dir)  # 多卡训练框架
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
    if_once_epoch = 5
    similar_altho = 3
    segmentation_altho = 0.5
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
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    guess_membership_data(p=(p, sub_p), model=model, dataset=dataset, accelerator=accelerator, save_path=save_path,
                          img_path=img_path,
                          dataset_choose=args.trainer.dataset, print_again=True)
    _, _ = eval_acc(p, target_model, (t_train, t_test), accelerator, shadow=False)
