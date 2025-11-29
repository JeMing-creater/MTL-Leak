import os
import cv2
import torch
import json
import shutil
import scipy.io as sio
import imageio
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil
import torch.nn.functional as F
import numpy as np
import torch.utils.data as Data
from torch.utils.data import random_split
from utils.utils import get_output, mkdir_if_missing


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top

# 作软标签生成存储
def giveSoftLabelPreprocessing(
        target_model, test_datasets, in_channels=3, in_size=32, save_path='./datasets/DataSet_store/',
        save_name='ResNet152_test_cifar10', device=torch.device('cuda')
):
    num = 0
    test_num = 0
    for _, _ in test_datasets:
        test_num += 1
    test_orginal_pic = np.zeros([test_num, in_channels, in_size, in_size])
    test_soft_label = np.zeros([test_num, 1])
    test_hard_label = np.zeros([test_num, 1])
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for img, label in test_datasets:
            img = img.type(torch.FloatTensor).to(device)
            label = label.type(torch.LongTensor).to(device)
            for x in img:
                x = x.unsqueeze(0)
                logits = target_model.to(device)(x)
                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                test_orginal_pic[num, :, :, :] = x.cpu().data.numpy()
                test_soft_label[num,] = pred.cpu().data.numpy()
            for lb in label:
                lb = lb.unsqueeze(0)
                test_hard_label[num,] = lb.cpu().data.numpy()
            num += 1
        np.save('{}/{}_pic.npy'.format(save_path, save_name), test_orginal_pic)
        np.save('{}/{}_soft_label.npy'.format(save_path, save_name), test_soft_label)
        np.save('{}/{}_hard_label.npy'.format(save_path, save_name), test_hard_label)
    print('Soft Labels has been preprocessed!')


def giveSoftdataPreprocessing(p, target_model, test_datasets, dataset='nyud', save_path='./datasets/Subdata/',
                              save_name='Hrnet18_test_nyud'):
    target_model.eval()
    stop = True
    i = 0
    if dataset == 'nyud':
        for i, batch in enumerate(test_datasets):
            if stop:
                images = batch['image'].cuda(non_blocking=True)
                targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
                out = target_model.cuda()(images)
                in_channels = images.real.shape[1]
                in_size1 = images.real.shape[2]
                in_size2 = images.real.shape[3]
                i1_channels = out['semseg'].real.shape[1]
                i1_size1 = out['semseg'].real.shape[2]
                i1_size2 = out['semseg'].real.shape[3]
                i2_channels = out['depth'].real.shape[1]
                i2_size1 = out['depth'].real.shape[2]
                i2_size2 = out['depth'].real.shape[3]
                i3_channels = targets['semseg'].real.shape[1]
                i3_size1 = targets['semseg'].real.shape[2]
                i3_size2 = targets['semseg'].real.shape[3]
                i4_channels = targets['depth'].real.shape[1]
                i4_size1 = targets['depth'].real.shape[2]
                i4_size2 = targets['depth'].real.shape[3]
                stop = False
        num = 0
        test_num = i + 1
        test_orginal_pic = np.zeros([test_num, in_channels, in_size1, in_size2])
        test_soft_label1 = np.zeros([test_num, i1_channels, i1_size1, i1_size2])
        test_soft_label2 = np.zeros([test_num, i2_channels, i2_size1, i2_size2])
        test_hard_label1 = np.zeros([test_num, i3_channels, i3_size1, i3_size2])
        test_hard_label2 = np.zeros([test_num, i4_channels, i4_size1, i4_size2])
        with torch.no_grad():
            for i, batch in enumerate(test_datasets):
                # Forward pass
                print('Preprocessing {} data'.format(i))
                images = batch['image'].cuda(non_blocking=True)
                targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
                for x in images:
                    x = x.unsqueeze(0)
                    logits = target_model.cuda()(x)
                    test_orginal_pic[num, :, :, :] = x.cpu().data.numpy()
                    test_soft_label1[num, :, :, :] = logits['semseg'].cpu().data.numpy()
                    test_soft_label2[num, :, :, :] = logits['depth'].cpu().data.numpy()
                for lb in targets['semseg']:
                    lb = lb.unsqueeze(0)
                    test_hard_label1[num, :, :, :] = lb.cpu().data.numpy()
                for lb in targets['depth']:
                    lb = lb.unsqueeze(0)
                    test_hard_label2[num, :, :, :] = lb.cpu().data.numpy()
                num += 1
            np.save('{}/{}_pic.npy'.format(save_path, save_name), test_orginal_pic)
            np.save('{}/{}_soft_label1.npy'.format(save_path, save_name), test_soft_label1)
            np.save('{}/{}_hard_label1.npy'.format(save_path, save_name), test_hard_label1)
            np.save('{}/{}_soft_label2.npy'.format(save_path, save_name), test_soft_label2)
            np.save('{}/{}_hard_label2.npy'.format(save_path, save_name), test_hard_label2)
    else:
        for i, batch in enumerate(test_datasets):
            if stop:
                images = batch['image'].cuda(non_blocking=True)
                targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
                out = target_model.cuda()(images)
                in_channels = images.real.shape[1]
                in_size1 = images.real.shape[2]
                in_size2 = images.real.shape[3]
                i1_channels = out['semseg'].real.shape[1]
                i1_size1 = out['semseg'].real.shape[2]
                i1_size2 = out['semseg'].real.shape[3]
                i2_channels = out['human_parts'].real.shape[1]
                i2_size1 = out['human_parts'].real.shape[2]
                i2_size2 = out['human_parts'].real.shape[3]
                i3_channels = out['sal'].real.shape[1]
                i3_size1 = out['sal'].real.shape[2]
                i3_size2 = out['sal'].real.shape[3]
                i4_channels = out['normals'].real.shape[1]
                i4_size1 = out['normals'].real.shape[2]
                i4_size2 = out['normals'].real.shape[3]
                t1_channels = targets['semseg'].real.shape[1]
                t1_size1 = targets['semseg'].real.shape[2]
                t1_size2 = targets['semseg'].real.shape[3]
                t2_channels = targets['human_parts'].real.shape[1]
                t2_size1 = targets['human_parts'].real.shape[2]
                t2_size2 = targets['human_parts'].real.shape[3]
                t3_channels = targets['sal'].real.shape[1]
                t3_size1 = targets['sal'].real.shape[2]
                t3_size2 = targets['sal'].real.shape[3]
                t4_channels = targets['normals'].real.shape[1]
                t4_size1 = targets['normals'].real.shape[2]
                t4_size2 = targets['normals'].real.shape[3]
                stop = False
        num = 0
        test_num = i + 1

        test_orginal_pic = np.zeros([test_num, in_channels, in_size1, in_size2])
        test_soft_label1 = np.zeros([test_num, i1_channels, i1_size1, i1_size2])
        test_soft_label2 = np.zeros([test_num, i2_channels, i2_size1, i2_size2])
        test_soft_label3 = np.zeros([test_num, i3_channels, i3_size1, i3_size2])
        test_soft_label4 = np.zeros([test_num, i4_channels, i4_size1, i4_size2])
        test_hard_label1 = np.zeros([test_num, t1_channels, t1_size1, t1_size2])
        test_hard_label2 = np.zeros([test_num, t2_channels, t2_size1, t2_size2])
        test_hard_label3 = np.zeros([test_num, t3_channels, t3_size1, t3_size2])
        test_hard_label4 = np.zeros([test_num, t4_channels, t4_size1, t4_size2])

        with torch.no_grad():
            for i, batch in enumerate(test_datasets):
                # Forward pass
                print('Preprocessing {} data'.format(i))
                images = batch['image'].cuda(non_blocking=True)
                targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
                for x in images:
                    x = x.unsqueeze(0)
                    logits = target_model.cuda()(x)
                    test_orginal_pic[num, :, :, :] = x.cpu().data.numpy()
                    test_soft_label1[num, :, :, :] = logits['semseg'].cpu().data.numpy()
                    test_soft_label2[num, :, :, :] = logits['human_parts'].cpu().data.numpy()
                    test_soft_label3[num, :, :, :] = logits['sal'].cpu().data.numpy()
                    test_soft_label4[num, :, :, :] = logits['normals'].cpu().data.numpy()
                for lb in targets['semseg']:
                    lb = lb.unsqueeze(0)
                    test_hard_label1[num, :, :, :] = lb.cpu().data.numpy()
                for lb in targets['human_parts']:
                    lb = lb.unsqueeze(0)
                    test_hard_label2[num, :, :, :] = lb.cpu().data.numpy()
                for lb in targets['sal']:
                    lb = lb.unsqueeze(0)
                    test_hard_label3[num, :, :, :] = lb.cpu().data.numpy()
                for lb in targets['normals']:
                    lb = lb.unsqueeze(0)
                    test_hard_label4[num, :, :, :] = lb.cpu().data.numpy()
                num += 1
            np.save('{}/{}_pic.npy'.format(save_path, save_name), test_orginal_pic)
            np.save('{}/{}_soft_label1.npy'.format(save_path, save_name), test_soft_label1)
            np.save('{}/{}_hard_label1.npy'.format(save_path, save_name), test_hard_label1)
            np.save('{}/{}_soft_label2.npy'.format(save_path, save_name), test_soft_label2)
            np.save('{}/{}_hard_label2.npy'.format(save_path, save_name), test_hard_label2)
            np.save('{}/{}_soft_label3.npy'.format(save_path, save_name), test_soft_label3)
            np.save('{}/{}_hard_label3.npy'.format(save_path, save_name), test_hard_label3)
            np.save('{}/{}_soft_label4.npy'.format(save_path, save_name), test_soft_label4)
            np.save('{}/{}_hard_label4.npy'.format(save_path, save_name), test_hard_label4)


def giveSoftdataPreprocessing_single(p, target_model, test_datasets, task_name='', save_path='./datasets/Subdata/',
                                     save_name='Hrnet18_test_nyud'):
    target_model.eval()
    stop = True
    i = 0
    for i, batch in enumerate(test_datasets):
        if stop:
            images = batch['image'].cuda(non_blocking=True)
            # images = batch['image'].cpu()
            targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
            # targets = {task: batch[task].cpu() for task in p.ALL_TASKS.NAMES}
            out = target_model.cuda()(images)
            in_channels = images.real.shape[1]
            in_size1 = images.real.shape[2]
            in_size2 = images.real.shape[3]
            i1_channels = out[task_name].real.shape[1]
            i1_size1 = out[task_name].real.shape[2]
            i1_size2 = out[task_name].real.shape[3]
            i3_channels = targets[task_name].real.shape[1]
            i3_size1 = targets[task_name].real.shape[2]
            i3_size2 = targets[task_name].real.shape[3]
            stop = False
    num = 0
    test_num = i + 1
    with torch.no_grad():
        test_orginal_pic = np.zeros([test_num, in_channels, in_size1, in_size2])
        for i, batch in enumerate(test_datasets):
            # Forward pass
            print('Preprocessing {} test_orginal_pic data'.format(i))
            images = batch['image'].cuda(non_blocking=True)
            # images = batch['image'].cpu()
            targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
            # targets = {task: batch[task].cpu() for task in p.ALL_TASKS.NAMES}
            for x in images:
                x = x.unsqueeze(0)
                test_orginal_pic[num, :, :, :] = x.cpu().data.numpy()
            num += 1
        np.save('{}/{}_pic.npy'.format(save_path, save_name), test_orginal_pic)
    with torch.no_grad():
        num = 0
        test_soft_label1 = np.zeros([test_num, i1_channels, i1_size1, i1_size2])
        for i, batch in enumerate(test_datasets):
            # Forward pass
            print('Preprocessing {} test_soft_label test_orginal_pic data'.format(i))
            images = batch['image'].cuda(non_blocking=True)
            # images = batch['image'].cpu()
            targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
            # targets = {task: batch[task].cpu() for task in p.ALL_TASKS.NAMES}
            for x in images:
                x = x.unsqueeze(0)
                logits = target_model.cuda()(x)
                test_soft_label1[num, :, :, :] = logits[task_name].cpu().data.numpy()
            num += 1
        np.save('{}/{}_soft_label1.npy'.format(save_path, save_name), test_soft_label1)
    with torch.no_grad():
        num = 0
        test_hard_label1 = np.zeros([test_num, i3_channels, i3_size1, i3_size2])
        for i, batch in enumerate(test_datasets):
            # Forward pass
            print('Preprocessing {} hard_label test_orginal_pic data'.format(i))
            # targets = {task: batch[task].cpu() for task in p.ALL_TASKS.NAMES}
            for lb in targets[task_name]:
                lb = lb.unsqueeze(0)
                test_hard_label1[num, :, :, :] = lb.cpu().data.numpy()
            num += 1
        np.save('{}/{}_hard_label1.npy'.format(save_path, save_name), test_hard_label1)


# 加载数据函数
def getData(data_path):
    def read_cam(path):
        with open(path, 'rb') as f:
            img2_array = np.load(f)
        return img2_array

    if 'pic' in data_path:
        print('read orginal_pic ...')
    elif 'soft_label' in data_path:
        print('read orginal_label ...')
    elif 'hard' in data_path:
        print('read camsave ...')
    else:
        print('data ...')
    data = read_cam(data_path).astype(np.float32)
    return data


# 提供训练替代模型的DataLoader
def giveSubstitutionData(dataset, pic_path, softlabel1_path, softlabel2_path, hardlabel1_path, hardlabel2_path,
                         softlabel3_path=None, softlabel4_path=None, hardlabel3_path=None, hardlabel4_path=None,
                         division=0.8, batch_size=1):
    orginal_pic_transforms = torch.from_numpy(getData(pic_path))
    orginal_soft_label1_transforms = torch.from_numpy(getData(softlabel1_path))
    orginal_hard_label1_transforms = torch.from_numpy(getData(hardlabel1_path))
    orginal_soft_label2_transforms = torch.from_numpy(getData(softlabel2_path))
    orginal_hard_label2_transforms = torch.from_numpy(getData(hardlabel2_path))

    if dataset == 'nyud':
        dataset = Data.TensorDataset(orginal_pic_transforms, orginal_soft_label1_transforms,
                                     orginal_hard_label1_transforms, orginal_soft_label2_transforms,
                                     orginal_hard_label2_transforms)
    else:
        orginal_soft_label3_transforms = torch.from_numpy(getData(softlabel3_path))
        orginal_hard_label3_transforms = torch.from_numpy(getData(hardlabel3_path))
        orginal_soft_label4_transforms = torch.from_numpy(getData(softlabel4_path))
        orginal_hard_label4_transforms = torch.from_numpy(getData(hardlabel4_path))
        dataset = Data.TensorDataset(orginal_pic_transforms, orginal_soft_label1_transforms,
                                     orginal_hard_label1_transforms, orginal_soft_label2_transforms,
                                     orginal_hard_label2_transforms, orginal_soft_label3_transforms,
                                     orginal_hard_label3_transforms, orginal_soft_label4_transforms,
                                     orginal_hard_label4_transforms)

    train_data, eval_data = random_split(
        dataset,
        [round(division * orginal_pic_transforms.shape[0]), round((1 - division) * orginal_pic_transforms.shape[0])],
        generator=torch.Generator().manual_seed(42)
    )
    train = Data.DataLoader(
        dataset=train_data,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
    )
    test = Data.DataLoader(
        dataset=eval_data,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
    )
    return train, test


def giveSubstitutionData_single(pic_path, softlabel1_path, hardlabel1_path, division=0.8, batch_size=1):
    orginal_pic_transforms = torch.from_numpy(getData(pic_path))
    orginal_soft_label1_transforms = torch.from_numpy(getData(softlabel1_path))
    orginal_hard_label1_transforms = torch.from_numpy(getData(hardlabel1_path))

    dataset = Data.TensorDataset(orginal_pic_transforms, orginal_soft_label1_transforms, orginal_hard_label1_transforms)

    train_data, eval_data = random_split(
        dataset,
        [round(division * orginal_pic_transforms.shape[0]), round((1 - division) * orginal_pic_transforms.shape[0])],
        generator=torch.Generator().manual_seed(42)
    )
    train = Data.DataLoader(
        dataset=train_data,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        drop_last=True
    )
    test = Data.DataLoader(
        dataset=eval_data,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        drop_last=True
    )
    return train, test


def give_sub_data(val_dataset, batch_size=1, num_workers=0, test_row=0.2):
    # 组合全部数据
    test_size = int(test_row * len(val_dataset))
    target_size = len(val_dataset) - test_size

    target_data, test_data = torch.utils.data.random_split(val_dataset,
                                                              [target_size,test_size])

    # dataloader
    tarin_dataloader = DataLoader(target_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                         num_workers=num_workers, collate_fn=collate_mil)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                       num_workers=num_workers, collate_fn=collate_mil)

    return tarin_dataloader,test_dataloader


def give_data_save(train, test,save_path='./datasets/Subdata/', save_name='multi_task_baseline'):
    def save_data(data, save_path, save_name):
        size = data.batch_sampler.sampler.num_samples
        list = []
        for i, batch in enumerate(data):
            list.append(batch)
        np.save('{}/{}_pic.npy'.format(save_path, save_name), list)
    save_data(train, save_path=save_path, save_name=save_name + '_train')
    save_data(test, save_path=save_path, save_name=save_name + '_test')
    print('all data save')

def give_all_data(model_name):
    def get_data(model_name, data_name):
        pic_path = '{}_{}_pic.npy'.format(model_name, data_name)
        data = np.load(pic_path, allow_pickle=True).tolist()
        return data
    train = get_data(model_name=model_name, data_name='train')
    test = get_data(model_name=model_name, data_name='test')
    return train, test

@torch.no_grad()
def save_model_predictions(p, val_loader, model):
    """ Save model predictions for all tasks """
    print('Save model predictions to {}'.format(p['save_dir']))
    model.eval()
    tasks = p.TASKS.NAMES
    save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks}
    for save_dir in save_dirs.values():
        mkdir_if_missing(save_dir)
    for task in p.TASKS.NAMES:
        shutil.rmtree(save_dirs[task])
        os.makedirs(save_dirs[task])
    # for ii, sample in enumerate(val_loader):
    # for ii, (img, label1, label2) in enumerate(val_loader):
    ii = 0
    for sample in val_loader:
        inputs, meta = sample['image'].cuda(non_blocking=True), sample['meta']
        img_size = (inputs.size(2), inputs.size(3))
        output = model(inputs)

        for task in p.TASKS.NAMES:
            output_task = get_output(output[task], task).cpu().data.numpy()
            for jj in range(int(inputs.size()[0])):
                if len(sample[task][jj].unique()) == 1 and sample[task][jj].unique() == 255:
                    continue
                fname = meta['image'][jj]
                t1 = output_task[jj]
                t2 = meta['im_size'][1][jj]
                t3 = meta['im_size'][0][jj]
                t4 = p.TASKS.INFER_FLAGVALS[task]
                result = cv2.resize(output_task[jj], dsize=(meta['im_size'][1][jj], meta['im_size'][0][jj]),
                                    interpolation=p.TASKS.INFER_FLAGVALS[task])
                if task == 'depth':
                    sio.savemat(os.path.join(save_dirs[task], fname + '.mat'), {'depth': result})
                else:
                    imageio.imwrite(os.path.join(save_dirs[task], fname + '.png'), result.astype(np.uint8))

        ii += 0


def calculate_multi_task_performance(eval_dict, single_task_dict):
    assert (set(eval_dict.keys()) == set(single_task_dict.keys()))
    tasks = eval_dict.keys()
    num_tasks = len(tasks)
    mtl_performance = 0.0

    for task in tasks:
        mtl = eval_dict[task]
        stl = single_task_dict[task]

        if task == 'depth':  # rmse lower is better
            mtl_performance -= (mtl['rmse'] - stl['rmse']) / stl['rmse']

        elif task in ['semseg', 'sal', 'human_parts']:  # mIoU higher is better
            mtl_performance += (mtl['mIoU'] - stl['mIoU']) / stl['mIoU']

        elif task == 'normals':  # mean error lower is better
            mtl_performance -= (mtl['mean'] - stl['mean']) / stl['mean']

        elif task == 'edge':  # odsF higher is better
            mtl_performance += (mtl['odsF'] - stl['odsF']) / stl['odsF']

        else:
            raise NotImplementedError

    return mtl_performance / num_tasks


def eval_all_results(p, definition_db=None):
    save_dir = p['save_dir']
    results = {}
    if 'semseg' in p.TASKS.NAMES:
        from utils.eval_semseg import eval_semseg_predictions
        results['semseg'] = eval_semseg_predictions(database=p['val_db_name'],
                                                    save_dir=save_dir, overfit=p.overfit, definition_db=definition_db)

    if 'depth' in p.TASKS.NAMES:
        from utils.eval_depth import eval_depth_predictions
        results['depth'] = eval_depth_predictions(database=p['val_db_name'],
                                                  save_dir=save_dir, overfit=p.overfit, definition_db=definition_db)

    if 'human_parts' in p.TASKS.NAMES:
        from utils.eval_human_parts import eval_human_parts_predictions
        results['human_parts'] = eval_human_parts_predictions(database=p['val_db_name'],
                                                              save_dir=save_dir, overfit=p.overfit,definition_db=definition_db)

    if 'sal' in p.TASKS.NAMES:
        from utils.eval_sal import eval_sal_predictions
        results['sal'] = eval_sal_predictions(database=p['val_db_name'],
                                              save_dir=save_dir, overfit=p.overfit,definition_db=definition_db)

    if 'normals' in p.TASKS.NAMES:
        from utils.eval_normals import eval_normals_predictions
        results['normals'] = eval_normals_predictions(database=p['val_db_name'],
                                                      save_dir=save_dir, overfit=p.overfit,definition_db=definition_db)


    if p['setup'] == 'multi_task':  # Perform the multi-task performance evaluation
        single_task_test_dict = {}
        for task, test_dict in p.TASKS.SINGLE_TASK_TEST_DICT.items():
            with open(test_dict, 'r') as f_:
                single_task_test_dict[task] = json.load(f_)
        results['multi_task_performance'] = calculate_multi_task_performance(results, single_task_test_dict)
        print('Multi-task learning performance on test set is %.2f' % (100 * results['multi_task_performance']))
    return results

