import os
import cv2
import yaml
import torch
import datetime
from objprint import objstr
from termcolor import colored
from easydict import EasyDict
import torch.nn.functional as F
from accelerate import Accelerator
from utils.config import create_config
from timm.optim import optim_factory
from utils.utils import AverageMeter, ProgressMeter, PerformanceMeter
from utils.common_config import get_train_dataset, get_transformations, \
    get_val_dataset, get_train_dataloader, get_val_dataloader, \
    get_optimizer, get_model, adjust_learning_rate, \
    get_criterion
from utils.evaluate_utils import validate_results,calculate_multi_task_performance,eval_all_results,get_output
from utils.utils import Logger
from utils.resum import load_pth
from utils.loss_functions import get_loss_meters
from Multi_data_division import division

# 配置读取
args = EasyDict(yaml.load(open('config_single.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))  # 读取配置
cv2.setNumThreads(0)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.trainer.card_ID


def resum(p, model, test, accelerator):
    # if p['resum'] != True:
    #     accelerator.print(colored('remove checkpoint', 'blue'))
    #     if os.path.exists(p['checkpoint']):
    #         os.remove(p['checkpoint'])
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
            if epoch + 1 > p['epochs'] - 3:
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
    if os.path.exists(p['best_model']):
        model_checkpoint = torch.load(p['best_model'])
    else:
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
        if 'semseg' in model_choose:
            exp = exp + 'semseg.yml'
        else:
            exp = exp + 'depth.yml'
    else:
        exp = './files/models/yml/pascal/'
        if 'hrnet18' in model_choose:
            exp = exp + 'hrnet18/'
        else:
            exp = exp + 'resnet18/'
        if 'semseg' in model_choose:
            exp = exp + 'semseg.yml'
        elif 'sal' in model_choose:
            exp = exp + 'sal.yml'
        elif 'human_parts' in model_choose:
            exp = exp + 'human_parts.yml'
        else:
            exp = exp + 'normals.yml'
    return exp


if __name__ == '__main__':
    logging_dir = os.getcwd() + '/logs/' + str(datetime.datetime.now())  # 生成logs记录文件文件夹
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], project_dir=logging_dir)  # 多卡训练框架
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
    save_name = 'MIA_{}'.format(p['dataset'])
    save_path = './datasets/MIAdata/'
    t_train, t_test, s_train, s_test, t_size, s_size = division(dataset=p['dataset'], save_name=save_name,
                                                                save_path=save_path)

    accelerator.print('Train transformations:')
    accelerator.print(train_transforms)
    accelerator.print('Val transformations:')
    accelerator.print(val_transforms)

    # 加载模型与数据
    target_model, shadow_model, optimizer, optimizer_s = accelerator.prepare(target_model, shadow_model, optimizer,
                                                                             optimizer_s)

    # Resume from checkpoint
    p['resum'] = args.trainer.resum
    sub_p['resum'] = args.trainer.resum
    target_model, start_epoch, best_result = resum(p, target_model, t_test, accelerator)
    shadow_model, s_start_epoch, s_best_result = resum(sub_p, shadow_model, s_test, accelerator)

    # 主模型/影模型 训练
    if args.trainer.resum != True:
        # 训练目标模型
        target_model = train_model(p, target_model, criterion, optimizer, (t_train, t_test), start_epoch, best_result,
                                   accelerator,
                                   False)
        #  查看性能
        # train_eval_stats, test_eval_stats = eval_acc(p, target_model, (t_train, t_test), accelerator, shadow=False)
        shadow_model = train_model(sub_p, shadow_model, criterion, optimizer_s, (s_train, s_test), s_start_epoch,
                                   s_best_result,
                                   accelerator,
                                   True)
        # s_train_eval_stats, s_test_eval_stats = eval_acc(sub_p, shadow_model, (s_train, s_test), accelerator,
        #                                                  shadow=True)

    #  查看性能
    _, _ = eval_acc(p, target_model, (t_train, t_test), accelerator, shadow=False)
    _, _ = eval_acc(sub_p, shadow_model, (s_train, s_test), accelerator, shadow=True)
