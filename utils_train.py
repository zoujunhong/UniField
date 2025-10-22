import sys
import torch
import os
import logging
import re
import functools
import fnmatch
import numpy as np

def load_model_weights(model, checkpoint_path):
    """
    Load model weights from a checkpoint, skipping incompatible layers.

    Args:
        model (torch.nn.Module): The PyTorch model to load weights into.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        model (torch.nn.Module): The model with loaded weights.
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    keys_list = list(checkpoint.keys())
    for key in keys_list:
        if '_orig_mod.' in key:
            deal_key = key.replace('_orig_mod.', '')
            checkpoint[deal_key] = checkpoint[key]
            del checkpoint[key]
    
    keys_list = list(checkpoint.keys())
    for key in keys_list:
        if 'module.' in key:
            deal_key = key.replace('module.', '')
            checkpoint[deal_key] = checkpoint[key]
            del checkpoint[key]
    # Get state dictionaries
    model_state_dict = model.state_dict()
    checkpoint_state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Filter out mismatched weights
    filtered_state_dict = {}
    for name, param in checkpoint_state_dict.items():
        if name in model_state_dict:
            if model_state_dict[name].shape == param.shape:
                filtered_state_dict[name] = param
            else:
                print(f"Skipping weight: {name} due to shape mismatch ({param.shape} vs {model_state_dict[name].shape})")
        else:
            print(f"Skipping weight: {name} as it is not in the model")

    # Update model state dict with filtered weights
    model_state_dict.update(filtered_state_dict)

    # Load updated state dict into the model
    model.load_state_dict(model_state_dict)

    return model

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def checkpoint(nets, args, epoch): # only save parameters that requires grad
    print('Saving checkpoints...')
    net_encoder = nets.module
    
    # to_save = OrderedDict()
    # for key in net_encoder.state_dict().keys():
    #     print(key, net_encoder.state_dict()[key])
    #     if net_encoder.state_dict()[key].requires_grad == True:
    #         print(key)
    #         to_save[key] = net_encoder.state_dict()[key]
    if not os.path.exists(args.saveroot):
        os.makedirs(args.saveroot, exist_ok=True)
        
    torch.save(
        net_encoder.state_dict(),
        '{}/model_epoch_{}.pth'.format(args.saveroot, epoch))

def get_common(list_,predlist,clip_num,h,w):
    accs = []
    for i in range(len(list_)-clip_num):
        global_common = np.ones((h,w))
        predglobal_common = np.ones((h,w))


        for j in range(1,clip_num):
            common = (list_[i] == list_[i+j])
            global_common = np.logical_and(global_common,common)
            pred_common = (predlist[i]==predlist[i+j])
            predglobal_common = np.logical_and(predglobal_common,pred_common)
        pred = (predglobal_common*global_common)

        acc = pred.sum()/global_common.sum()
        accs.append(acc)
    return accs

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def beforeval(self):
        isval = np.sum(self.confusion_matrix,axis=1)>0
        self.confusion_matrix = self.confusion_matrix*isval

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc


    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        isval = np.sum(self.confusion_matrix,axis=1)>0
        MIoU = np.nansum(MIoU*isval)/isval.sum()
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        #print(mask)
        #print(gt_image.shape)
        #print(gt_image[mask])
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
#        print(label.shape)
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def setup_logger(distributed_rank=0, filename="log.txt"):
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    return logger


def find_recursive(root_dir, ext='.jpg'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            if filename[0]=='.':
                continue
            files.append(os.path.join(root, filename))
    return files


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        # self.avg = 0.98 * self.avg + 0.02 * val
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


class NotSupportedCliException(Exception):
    pass


def process_range(xpu, inp):
    start, end = map(int, inp)
    if start > end:
        end, start = start, end
    return map(lambda x: '{}{}'.format(xpu, x), range(start, end+1))


REGEX = [
    (re.compile(r'^gpu(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^gpu(\d+)-(?:gpu)?(\d+)$'),
     functools.partial(process_range, 'gpu')),
    (re.compile(r'^(\d+)-(\d+)$'),
     functools.partial(process_range, 'gpu')),
]


def parse_devices(input_devices):

    """Parse user's devices input str to standard format.
    e.g. [gpu0, gpu1, ...]

    """
    ret = []
    for d in input_devices.split(','):
        for regex, func in REGEX:
            m = regex.match(d.lower().strip())
            if m:
                tmp = func(m.groups())
                # prevent duplicate
                for x in tmp:
                    if x not in ret:
                        ret.append(x)
                break
        else:
            raise NotSupportedCliException(
                'Can not recognize device: "{}"'.format(d))
    return ret

def get_params_groups(model, lr=0.001, wd=0.01, 
                      special_keyword=[], special_lr=[]):

    assert len(special_keyword) == len(special_lr), \
        "special_keyword and special_lr must have the same length"

    # 存储已经匹配了特殊 keyword 的 param，防止重复添加
    matched_params = set()

    # 参数组结果
    param_groups = []

    # 添加特殊 keyword 的分组
    for keyword, sk_lr in zip(special_keyword, special_lr):
        special_regularized = []
        special_not_regularized = []

        for name, param in model.named_parameters():
            if not param.requires_grad or param in matched_params:
                continue

            if keyword in name:
                matched_params.add(param)
                if name.endswith(".bias") or len(param.shape) == 1:
                    special_not_regularized.append(param)
                else:
                    special_regularized.append(param)

        if special_regularized:
            param_groups.append({
                'params': special_regularized,
                'base_lr': sk_lr,
                'weight_decay': wd
            })
        if special_not_regularized:
            param_groups.append({
                'params': special_not_regularized,
                'base_lr': sk_lr,
                'weight_decay': 0.
            })

    # 添加剩余的常规参数
    regularized = []
    not_regularized = []

    for name, param in model.named_parameters():
        if not param.requires_grad or param in matched_params:
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)

    if regularized:
        param_groups.append({
            'params': regularized,
            'base_lr': lr,
            'weight_decay': wd
        })
    if not_regularized:
        param_groups.append({
            'params': not_regularized,
            'base_lr': lr,
            'weight_decay': 0.
        })

    return param_groups

def get_finetune_param_groups(model, wd=0.01, 
                              special_keyword=None, special_lr=None):
    if special_keyword is None:
        special_keyword = []
    if special_lr is None:
        special_lr = []

    assert len(special_keyword) == len(special_lr), \
        "special_keyword and special_lr must have the same length"

    param_groups = []
    matched_params = set()
    trainable_modules = set()  # 用于记录需要 train 的模块

    for keyword, sk_lr in zip(special_keyword, special_lr):
        special_regularized = []
        special_not_regularized = []

        for name, param in model.named_parameters():
            if param in matched_params:
                continue

            if keyword in name:
                matched_params.add(param)
                param.requires_grad = True  # 开启梯度

                # 找到对应模块名，标记为需要 train
                module_name = name.rsplit('.', 1)[0]  # 去掉 .weight / .bias
                trainable_modules.add(module_name)

                if name.endswith(".bias") or len(param.shape) == 1:
                    special_not_regularized.append(param)
                else:
                    special_regularized.append(param)

        if special_regularized:
            param_groups.append({
                'params': special_regularized,
                'base_lr': sk_lr,
                'weight_decay': wd
            })
        if special_not_regularized:
            param_groups.append({
                'params': special_not_regularized,
                'base_lr': sk_lr,
                'weight_decay': 0.
            })

    # 对其余参数，关闭 requires_grad
    for name, param in model.named_parameters():
        if param not in matched_params:
            param.requires_grad = False

    # 设置模块状态
    for name, module in model.named_modules():
        if name in trainable_modules:
            module.train()
        else:
            module.eval()

    return param_groups

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_iters=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def exp_scheduler(base_value, decay_rate, decay_steps, epochs, niter_per_ep, warmup_iters=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value * (decay_rate**(warmup_iters / decay_steps)), warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = base_value * (decay_rate**(iters / decay_steps))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


