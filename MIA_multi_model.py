import os
import csv
import cv2
import yaml
import torch
import datetime
from torch import optim
import matplotlib.pyplot as plt
import math
from torch import nn
import numpy as np
from objprint import objstr
from termcolor import colored
from easydict import EasyDict
from accelerate import Accelerator
from utils.config import create_config
from timm.optim import optim_factory
from utils.utils import ProgressMeter, PerformanceMeter
from utils.common_config import get_transformations, \
    get_optimizer, get_model, adjust_learning_rate, \
    get_criterion
from utils.evaluate_utils import validate_results, eval_all_results, \
    get_output, SemsegMeter, DepthMeter, HumanPartsMeter, SaliencyMeter, NormalsMeter
from utils.utils import Logger
from utils.resum import load_pth
from utils.loss_functions import get_loss_meters
from Multi_data_division import division
from utils.MIAutils import get_resnet_classification_model
from files.models.classification_model import ClassFormer
import pickle
import random
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import torch.nn.functional as F

# 配置读取
args = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))  # 读取配置
cv2.setNumThreads(0)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.trainer.card_ID


def train_model(p, model, criterion, optimizer, data, start_epoch, best_result, accelerator, shadow=False):
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
        for batch in train:
            # for i, batch in enumerate(ttrain):
            # Forward pass
            images = batch['image'].to(accelerator.device)
            targets = {task: batch[task].to(accelerator.device) for task in p.ALL_TASKS.NAMES}
            output = model(images)

            # Measure loss and performance
            loss_dict = criterion(output, targets)
            # out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in self.tasks]))
            for k, v in loss_dict.items():
                losses[k].update(v.item())
            performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES},
                                     {t: targets[t] for t in p.TASKS.NAMES})

            # Backward
            optimizer.zero_grad()
            accelerator.backward(loss_dict['total'])
            # loss_dict['total'].backward()
            optimizer.step()

            if i % 25 == 0:
                progress.display(i)
            i += 1
        # Evaluate
        # 此处本该为最后轮作验证，但由于验证过于耗时，改为最后10轮验证
        eval_bool = False
        if p['10_eval'] == True:
            if epoch + 1 > p['epochs'] - 10:
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
    train, test = dataset
    input_tensors = []
    label_tensors = []
    with torch.no_grad():
        for i in range(0, len(test)):
            batch = train[i]
            non_batch = test[i]
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

            # add list
            input_tensors.append(diff_input.to('cpu'))
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
                for i in range(len(data1)):
                    writer.writerow([i, data1[i], data2[i], data3[i], data4[i], data5[i], data6[i], data7[i], data8[i]])
    else:
        data1, data2 = data
        img_path = img_path + '/' + 'diff.csv'
        with open(img_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Mem_Diff', 'Non_Mem_Diff'])  # 写入表头
            for i in range(len(data1)):
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


def train_attack_model(p, data, accelerator, dataset_choose='nyud', attack_alptho=3):
    (train_input_tensor, train_label_tensor), (test_input_tensor, test_label_tensor) = data
    attack_model = get_resnet_classification_model(p, arch="resnet18", input_channel=1, num_classes=2,
                                                   dilated=True)
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
            input = get_attack_model_input(tr_input_tensor, dataset_choose, attack_alptho)
            optimizer.zero_grad()
            output = attack_model(input)
            loss = criterion(output, tr_label_tensor)
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
        all_loss = all_loss / train_input_tensor.size(0)
        train_acc = test_attack_model(attack_model, (train_input_tensor, train_label_tensor), accelerator,
                                      dataset_choose, attack_alptho)
        test_acc = test_attack_model(attack_model, (test_input_tensor, test_label_tensor), accelerator, dataset_choose,
                                     attack_alptho)
        if best_acc < test_acc:
            best_acc = test_acc
        accelerator.print(
            f'[{epoch + 1}/{30}] Best attack acc: {best_acc * 100} %  Now Train acc: {train_acc * 100}% Test acc: {test_acc * 100}% Loss:{all_loss}')
    accelerator.print(f'Attack acc: {best_acc * 100} %')
    return best_acc


def test_attack_model(attack_model, data, accelerator, dataset_choose='nyud', attack_alptho=3):
    (test_input_tensor, test_label_tensor) = data
    attack_model.eval()
    acc = 0
    for i in range(0, test_input_tensor.size(0)):
        tr_input_tensor = test_input_tensor[i].type(torch.FloatTensor).to(accelerator.device)
        tr_label_tensor = test_label_tensor[i].type(torch.LongTensor).to(accelerator.device)
        input = get_attack_model_input(tr_input_tensor, dataset_choose, attack_alptho)
        # input = torch.cat((orimage_tensor, semseg_tensor, depth_tensor), dim=1)
        output = attack_model(input).argmax(dim=1)
        acc += torch.eq(output, tr_label_tensor).float().sum().item() / output.size(0)
    return acc / test_input_tensor.size(0)


def guess(datasets, accelerator, alpth=2, dataset_choose='nyud', img_path='.csv'):
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
        hmmeter = HumanPartsMeter(database='PASCALContext')
        hmmeter.update(pred, gt)
        acc = hmmeter.get_score(verbose=False)['mIoU']
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
        nometer = NormalsMeter()
        nometer.update(pred, gt)
        acc = nometer.get_score(verbose=False)['mean']
        return acc

    def give_diff_test(data, diff_alpth=2):
        # seg_mask = seg_tensor[0]
        # dep_map = dep_tensor[0]
        def diff_c(tensor1, tensor2):
            return torch.exp(diff_alpth * torch.relu(torch.abs(tensor1) - torch.abs(tensor2))).mean()

        diff_list = []
        for data_item in data:
            diff_list += torch.unsqueeze(torch.sum(data_item, dim=1) / data_item.size(1), dim=1)

        differences = []
        count = 0
        if len(diff_list) >= 2:
            for i in range(len(diff_list) - 1):
                for j in range(i + 1, len(diff_list)):
                    difference = diff_c(diff_list[i], diff_list[j]).item()
                    differences.append(difference)
                    count += 1
            diff_guess = torch.tensor(sum(differences) / count)
        else:
            diff_guess = torch.tensor(0)
        # semseg_tensor = torch.unsqueeze(torch.sum(seg_tensor, dim=1) / seg_tensor.size(1), dim=1)
        # depth_tensor = torch.unsqueeze(torch.sum(dep_tensor, dim=1) / dep_tensor.size(1), dim=1)
        #
        # diff_guess = diff_c(semseg_tensor, depth_tensor)
        # diff_guess = torch.tensor(math.exp(2 * object_consistency))
        return diff_guess

    def get_diff_threshold(high_data_array, low_data_array):
        def calculate_error(high_data_array, low_data_array, current_threshold):
            get_tr_once = 0
            get_te_once = 0
            for high_data in high_data_array:
                if high_data > current_threshold:
                    get_tr_once += 1
            for low_data in low_data_array:
                if low_data < current_threshold:
                    get_te_once += 1

            get_data_score = get_tr_once / len(high_data_array) + get_te_once / len(low_data_array)
            error = (2 - get_data_score) / 2
            return error

        min_error = 1
        threshold = 0
        for high_data in high_data_array:
            for low_data in low_data_array:
                current_threshold = (high_data + low_data) / 2

                error = calculate_error(high_data_array, low_data_array, current_threshold)

                # 更新阈值和误差值
                if error < min_error:
                    threshold = current_threshold
                    min_error = error
        return threshold

        # return diff_guess

    shadow_data, target_data = datasets
    train_input_tensor, train_label_tensor = shadow_data
    test_input_tensor, test_label_tensor = target_data

    if dataset_choose == 'nyud':
        tr_diff_list = []
        te_diff_list = []

        tr_seg_list = []
        tr_dep_list = []
        te_seg_list = []
        te_dep_list = []

        # 判断分割阈值
        with torch.no_grad():
            for i in range(0, train_input_tensor.size(0)):
                tr_input_tensor = train_input_tensor[i].type(torch.FloatTensor).to(accelerator.device)
                tr_label_tensor = train_label_tensor[i].type(torch.LongTensor).to(accelerator.device)
                tr_orimage_tensor, tr_semseg_tensor, tr_depth_tensor, tr_target_tensor = give_data(tr_input_tensor,
                                                                                                   dataset_choose)

                for j in range(0, tr_label_tensor.size(0)):
                    tst = torch.unsqueeze(tr_semseg_tensor[j], dim=0)
                    tstt = torch.unsqueeze(tr_target_tensor[:, 0:1, :, :][j], dim=0)
                    trt = torch.unsqueeze(tr_depth_tensor[j], dim=0)
                    trtt = torch.unsqueeze(tr_target_tensor[:, 1:2, :, :][j], dim=0)

                    diff_guess = give_diff_test((tst, trt), alpth)
                    # diff_guess = torch.mean(torch.abs(torch.abs(semseg_tensor) - depth_tensor))
                    seg_acc = give_semage_test(tst, tstt)
                    dep_acc = give_depth_test(trt, trtt)
                    membership_label = tr_label_tensor[j]

                    if membership_label == 1:
                        tr_diff_list.append(diff_guess.item())
                        tr_seg_list.append(seg_acc.item())
                        tr_dep_list.append(dep_acc.item())
                    else:
                        te_diff_list.append(diff_guess.item())
                        te_seg_list.append(seg_acc.item())
                        te_dep_list.append(dep_acc.item())

        import statistics
        tr_avg_threshold = sum(tr_diff_list) / len(tr_diff_list)
        te_avg_threshold = sum(te_diff_list) / len(te_diff_list)

        tr_max_diff_guess = max(tr_diff_list)
        te_max_diff_guess = max(te_diff_list)
        tr_min_diff_guess = min(tr_diff_list)
        te_min_diff_guess = min(te_diff_list)

        # normalized_value = (x - min_value) / (max_value - min_value)

        tr_median = statistics.median(tr_diff_list)
        te_median = statistics.median(te_diff_list)

        avg_threshold = get_diff_threshold(tr_diff_list, te_diff_list)

        membership_seg_threshold = max(te_seg_list)
        membership_dep_threshold = min(te_dep_list)

        attack_acc1 = 0
        attack_acc2 = 0

        # 作画
        pring_img(data=(tr_diff_list, te_diff_list), mood='diff', img_path=img_path)
        pring_img(data=(te_seg_list, te_dep_list, tr_seg_list, tr_dep_list), mood='data', img_path=img_path)
        with torch.no_grad():
            for i in range(0, test_input_tensor.size(0)):
                te_input_tensor = test_input_tensor[i].type(torch.FloatTensor).to(accelerator.device)
                te_label_tensor = test_label_tensor[i].type(torch.LongTensor).to(accelerator.device)
                te_orimage_tensor, te_semseg_tensor, te_depth_tensor, te_target_tensor = give_data(te_input_tensor,
                                                                                                   dataset_choose)

                guess_label_tensor1 = torch.zeros(te_label_tensor.size())
                guess_label_tensor2 = torch.zeros(te_label_tensor.size())
                for j in range(0, te_label_tensor.size(0)):
                    tst = torch.unsqueeze(te_semseg_tensor[j], dim=0)
                    tstt = torch.unsqueeze(te_target_tensor[:, 0:1, :, :][j], dim=0)
                    trt = torch.unsqueeze(te_depth_tensor[j], dim=0)
                    trtt = torch.unsqueeze(te_target_tensor[:, 1:2, :, :][j], dim=0)
                    diff_guess = give_diff_test((tst, trt), alpth)
                    te_seg_acc = give_semage_test(tst, tstt)
                    te_dep_acc = give_depth_test(trt, trtt)

                    # diff_guess = torch.mean(torch.abs(torch.abs(semseg_tensor) - depth_tensor))
                    if te_seg_acc > membership_seg_threshold or te_dep_acc < membership_dep_threshold:
                        guess_label_tensor1[j] = 1.

                    if diff_guess > te_max_diff_guess or diff_guess < te_min_diff_guess:
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
        return attack_acc1, attack_acc2
    else:
        tr_diff_list = []
        te_diff_list = []

        tr_seg_list = []
        tr_hp_list = []
        tr_sal_list = []
        tr_norm_list = []

        te_seg_list = []
        te_hp_list = []
        te_sal_list = []
        te_norm_list = []

        # 判断分割阈值
        with torch.no_grad():
            for i in range(0, train_input_tensor.size(0)):
                tr_input_tensor = train_input_tensor[i].type(torch.FloatTensor).to(accelerator.device)
                tr_label_tensor = train_label_tensor[i].type(torch.LongTensor).to(accelerator.device)
                tr_orimage_tensor, tr_semseg_tensor, tr_hp_tensor, tr_sal_tensor, tr_norm_tensor, tr_target_tensor = give_data(
                    tr_input_tensor, dataset_choose)

                for j in range(0, tr_label_tensor.size(0)):
                    s = torch.unsqueeze(tr_semseg_tensor[j], dim=0)
                    st = torch.unsqueeze(tr_target_tensor[:, 0:1, :, :][j], dim=0)
                    hp = torch.unsqueeze(tr_hp_tensor[j], dim=0)
                    hpt = torch.unsqueeze(tr_target_tensor[:, 1:2, :, :][j], dim=0)
                    sal = torch.unsqueeze(tr_sal_tensor[j], dim=0)
                    salt = torch.unsqueeze(tr_target_tensor[:, 2:3, :, :][j], dim=0)
                    norm = torch.unsqueeze(tr_norm_tensor[j], dim=0)
                    normt = torch.unsqueeze(tr_target_tensor[:, 3:6, :, :][j], dim=0)

                    diff_guess = give_diff_test((s, hp, sal, norm), alpth)

                    seg_acc = give_semage_test(s, st)
                    hp_acc = give_human_test(hp, hpt)
                    sal_acc = give_sal_test(sal, salt)
                    norm_acc = give_norm_test(norm, normt)

                    membership_label = tr_label_tensor[j]

                    if membership_label == 1:
                        tr_diff_list.append(diff_guess.item())
                        tr_seg_list.append(seg_acc.item())
                        tr_hp_list.append(hp_acc.item())
                        tr_sal_list.append(sal_acc)
                        tr_norm_list.append(norm_acc)
                    else:
                        te_diff_list.append(diff_guess.item())
                        te_seg_list.append(seg_acc.item())
                        te_hp_list.append(hp_acc.item())
                        te_sal_list.append(sal_acc)
                        te_norm_list.append(norm_acc)

        import statistics
        te_max_diff_guess = max(te_diff_list)
        te_min_diff_guess = min(te_diff_list)

        # normalized_value = (x - min_value) / (max_value - min_value)

        # avg_threshold = te_avg_threshold
        membership_seg_threshold = max(te_seg_list)
        membership_hp_threshold = max(te_hp_list)
        membership_sal_threshold = max(te_seg_list)
        membership_norm_threshold = min(te_norm_list)

        attack_acc1 = 0
        attack_acc2 = 0

        # 作画
        pring_img(data=(tr_diff_list, te_diff_list), mood='diff', img_path=img_path, dataset_choose=dataset_choose)
        pring_img(data=(
            te_seg_list, te_hp_list, te_sal_list, te_norm_list, tr_seg_list, tr_hp_list, tr_sal_list, tr_norm_list),
            mood='data', img_path=img_path, dataset_choose=dataset_choose)
        with torch.no_grad():
            for i in range(0, test_input_tensor.size(0)):
                te_input_tensor = test_input_tensor[i].type(torch.FloatTensor).to(accelerator.device)
                te_label_tensor = test_label_tensor[i].type(torch.LongTensor).to(accelerator.device)
                te_orimage_tensor, te_semseg_tensor, te_hp_tensor, te_sal_tensor, te_norm_tensor, te_target_tensor = give_data(
                    te_input_tensor, dataset_choose)
                guess_label_tensor1 = torch.zeros(te_label_tensor.size())
                guess_label_tensor2 = torch.zeros(te_label_tensor.size())

                for j in range(0, te_label_tensor.size(0)):
                    s = torch.unsqueeze(te_semseg_tensor[j], dim=0)
                    st = torch.unsqueeze(te_target_tensor[:, 0:1, :, :][j], dim=0)
                    hp = torch.unsqueeze(te_hp_tensor[j], dim=0)
                    hpt = torch.unsqueeze(te_target_tensor[:, 1:2, :, :][j], dim=0)
                    sal = torch.unsqueeze(te_sal_tensor[j], dim=0)
                    salt = torch.unsqueeze(te_target_tensor[:, 2:3, :, :][j], dim=0)
                    norm = torch.unsqueeze(te_norm_tensor[j], dim=0)
                    normt = torch.unsqueeze(te_norm_tensor[:, 3:6, :, :][j], dim=0)

                    diff_guess = give_diff_test((s, hp, sal, norm), alpth)

                    seg_acc = give_semage_test(s, st)
                    hp_acc = give_human_test(hp, hpt)
                    sal_acc = give_sal_test(sal, salt)
                    norm_acc = give_norm_test(norm, normt)

                    # diff_guess = torch.mean(torch.abs(torch.abs(semseg_tensor) - depth_tensor))
                    if seg_acc > membership_seg_threshold or hp_acc > membership_hp_threshold or sal_acc > membership_sal_threshold or norm_acc < membership_norm_threshold:
                        guess_label_tensor1[j] = 1.

                    if diff_guess > te_max_diff_guess or diff_guess < te_min_diff_guess:
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
        return attack_acc1, attack_acc2


def guess_membership_data(p, model, dataset, accelerator, save_path, img_path, alpth=2, dataset_choose='nyud',
                          print_again=False):
    p, sub_p = p
    target_model, shadow_model = model
    target_model, shadow_model = accelerator.prepare(target_model, shadow_model)
    (t_train, t_test), (s_train, s_test) = dataset
    if print_again == True or not (
            os.path.exists(save_path + '_shadow.npz') or os.path.exists(save_path + '_target.npz')):
        # shadow
        print_label(shadow_model, (s_train, s_test), accelerator, save_path + '_shadow.npz')
        # target
        print_label(target_model, (t_train, t_test), accelerator, save_path + '_target.npz')

    # load data
    # shadow
    shadow_data = load_label_data(save_path + '_shadow.npz')
    # target
    target_data = load_label_data(save_path + '_target.npz')

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
    best_label_acc, best_diff_acc = guess((shadow_data, target_data), dataset_choose=dataset_choose, img_path=img_path,
                                          alpth=alpth,
                                          accelerator=accelerator)
    # 打印
    accelerator.print(f'best_AMI_acc: {best_attack_model_acc * 100} %')
    accelerator.print(f'best_TI_acc: {best_label_acc * 100} %')


def load_model(p, model, accelerator):
    # model_checkpoint = torch.load(p['checkpoint'])['model']
    accelerator.print('Load Model ...')
    try:
        model_checkpoint = torch.load(p['best_model'])
    except:
        model_checkpoint = torch.load(p['checkpoint'])['model']
    model_checkpoint = load_pth(p, model_checkpoint)
    model.load_state_dict(model_checkpoint)
    model = accelerator.prepare(model)
    return model


if __name__ == '__main__':
    logging_dir = os.getcwd() + '/logs/' + str(datetime.datetime.now())  # 生成logs记录文件文件夹
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], logging_dir=logging_dir)  # 多卡训练框架
    Logger(logging_dir if accelerator.is_local_main_process else None)  # 用以产生训练记录文件
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(args))  # 打印参数信息

    # 提供训练指导
    config_exp = give_config_exp(args.trainer.dataset, args.trainer.model_choose)

    p = create_config(args.trainer.root_dir + '/target_results', config_exp)
    sub_p = create_config(args.trainer.root_dir + '/shadow_results', config_exp)
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

    # CUDNN
    accelerator.print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

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
    target_model, shadow_model = accelerator.prepare(target_model, shadow_model)

    # 查看性能
    # _, _ = eval_acc(p, target_model, (t_train, t_test), accelerator, shadow=False)
    # _, _ = eval_acc(sub_p, shadow_model, (s_train, s_test), accelerator, shadow=True)

    target_model = load_model(p, target_model, accelerator)
    shadow_model = load_model(sub_p, shadow_model, accelerator)

    # 攻击模型定义
    model = (target_model, shadow_model)
    dataset = ((t_train, t_test), (s_train, s_test))
    save_name = 'MIA_{}_original'.format(p['dataset'])
    save_path = './datasets/MIAdata/{}/'.format(args.trainer.model_choose)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path + save_name
    img_path = './files/data_details/{}_{}_original/'.format(p['dataset'], args.trainer.model_choose)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    guess_membership_data(p=(p, sub_p), model=model, dataset=dataset, accelerator=accelerator, save_path=save_path,
                          img_path=img_path, alpth=2,
                          dataset_choose=args.trainer.dataset, print_again=False)

