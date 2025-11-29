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
from utils.evaluate_utils import calculate_multi_task_performance

# 配置读取
args = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))  # 读取配置
cv2.setNumThreads(0)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.trainer.card_ID


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
        pring_img(data=(te_seg_list, te_dep_list, tr_seg_list, tr_dep_list), mood='data', img_path=img_path,
                  dataset_choose=dataset_choose)
    else:
        membership_seg_threshold = max(te_seg_list)
        membership_hp_threshold = max(te_hp_list)
        membership_sal_threshold = max(te_sal_list)
        membership_norm_threshold = min(te_norm_list)
        pring_img(data=(
            te_seg_list, te_hp_list, te_sal_list, te_norm_list, tr_seg_list, tr_hp_list, tr_sal_list,
            tr_norm_list),
            mood='data', img_path=img_path, dataset_choose=dataset_choose)
    membership_score_threshold = max(te_mu_score)
    pring_img(data=(tr_mu_score, te_mu_score), mood='score', img_path=img_path, dataset_choose=dataset_choose)
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

    img_path = './files/data_details/{}_{}_original/'.format(p['dataset'], args.trainer.model_choose)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    ami_acc = AMI(p, model, dataset, accelerator, attack_alptho=1)
    ti_acc = TI(p, model, dataset, accelerator, img_path)
    accelerator.print(f'AMI-MTL acc: {ami_acc * 100} %')
    accelerator.print(f'TI-MTL acc: {ti_acc * 100} %')
