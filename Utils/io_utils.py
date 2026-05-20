import os  # 操作系统相关功能
import sys  # 系统相关参数和函数
import yaml  # 处理yaml配置文件
import json  # 处理json文件
import torch  # PyTorch库
import random  # 随机数生成
import warnings  # 警告信息
import importlib  # 动态导入模块
import numpy as np  # 数值计算库


# 加载yaml配置文件
def load_yaml_config(path):
    with open(path) as f:  # 打开文件
        config = yaml.full_load(f)  # 读取yaml内容
    return config  # 返回配置字典


# 保存配置到yaml文件
def save_config_to_yaml(config, path):
    assert path.endswith('.yaml')  # 确保文件后缀为.yaml
    with open(path, 'w') as f:  # 打开文件以写入
        f.write(yaml.dump(config))  # 写入yaml内容
        f.close()  # 关闭文件


# 保存字典到json文件
def save_dict_to_json(d, path, indent=None):
    json.dump(d, open(path, 'w'), indent=indent)  # 直接写入json


# 从json文件加载字典
def load_dict_from_json(path):
    return json.load(open(path, 'r'))  # 读取json并返回


# 将命令行参数写入文件
def write_args(args, path):
    args_dict = dict((name, getattr(args, name)) for name in dir(args)if not name.startswith('_'))  # 获取参数字典
    with open(path, 'a') as args_file:  # 追加写入
        args_file.write('==> torch version: {}\n'.format(torch.__version__))  # 写入torch版本
        args_file.write('==> cudnn version: {}\n'.format(torch.backends.cudnn.version()))  # 写入cudnn版本
        args_file.write('==> Cmd:\n')  # 写入命令行
        args_file.write(str(sys.argv))
        args_file.write('\n==> args:\n')  # 写入参数
        for k, v in sorted(args_dict.items()):
            args_file.write('  %s: %s\n' % (str(k), str(v)))
        args_file.close()  # 关闭文件


# 设置全局随机种子
def seed_everything(seed, cudnn_deterministic=False):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    
    Args:
        seed: the integer value seed for global random state
    """
    if seed is not None:
        print(f"Global seed set to {seed}")  # 打印设置的种子
        random.seed(seed)  # 设置python随机种子
        np.random.seed(seed)  # 设置numpy随机种子
        torch.manual_seed(seed)  # 设置torch随机种子
        torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
        torch.backends.cudnn.deterministic = False  # 关闭cudnn确定性

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True  # 开启cudnn确定性
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')  # 警告信息


# 合并命令行opts到配置字典
def merge_opts_to_config(config, opts):
    def modify_dict(c, nl, v):
        if len(nl) == 1:
            c[nl[0]] = type(c[nl[0]])(v)  # 修改单层key
        else:
            # print(nl)
            c[nl[0]] = modify_dict(c[nl[0]], nl[1:], v)  # 递归修改多层key
        return c

    if opts is not None and len(opts) > 0:
        assert len(opts) % 2 == 0, "each opts should be given by the name and values! The length shall be even number!"
        for i in range(len(opts) // 2):
            name = opts[2*i]
            value = opts[2*i+1]
            config = modify_dict(config, name.split('.'), value)  # 修改配置
    return config


# 调试用：修改配置为单线程和batch=1
def modify_config_for_debug(config):
    config['dataloader']['num_workers'] = 0  # 设置线程数为0
    config['dataloader']['batch_size'] = 1  # 设置batch为1
    return config


# 获取模型参数信息
def get_model_parameters_info(model):
    # for mn, m in model.named_modules():
    parameters = {'overall': {'trainable': 0, 'non_trainable': 0, 'total': 0}}  # 初始化参数统计
    for child_name, child_module in model.named_children():  # 遍历模型子模块
        parameters[child_name] = {'trainable': 0, 'non_trainable': 0}
        for pn, p in child_module.named_parameters():  # 遍历参数
            if p.requires_grad:
                parameters[child_name]['trainable'] += p.numel()  # 可训练参数数
            else:
                parameters[child_name]['non_trainable'] += p.numel()  # 不可训练参数数
        parameters[child_name]['total'] = parameters[child_name]['trainable'] + parameters[child_name]['non_trainable']
        
        parameters['overall']['trainable'] += parameters[child_name]['trainable']
        parameters['overall']['non_trainable'] += parameters[child_name]['non_trainable']
        parameters['overall']['total'] += parameters[child_name]['total']
    
    # 格式化数字显示
    def format_number(num):
        K = 2**10
        M = 2**20
        G = 2**30
        if num > G: # K
            uint = 'G'
            num = round(float(num)/G, 2)
        elif num > M:
            uint = 'M'
            num = round(float(num)/M, 2)
        elif num > K:
            uint = 'K'
            num = round(float(num)/K, 2)
        else:
            uint = ''
        
        return '{}{}'.format(num, uint)
    
    # 递归格式化字典
    def format_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                format_dict(v)
            else:
                d[k] = format_number(v)
    
    format_dict(parameters)  # 格式化参数统计
    return parameters  # 返回参数信息


# 格式化秒为可读时间
def format_seconds(seconds):
    h = int(seconds // 3600)
    m = int(seconds // 60 - h * 60)
    s = int(seconds % 60)

    d = int(h // 24)
    h = h - d * 24

    if d == 0:
        if h == 0:
            if m == 0:
                ft = '{:02d}s'.format(s)
            else:
                ft = '{:02d}m:{:02d}s'.format(m, s)
        else:
           ft = '{:02d}h:{:02d}m:{:02d}s'.format(h, m, s)
 
    else:
        ft = '{:d}d:{:02d}h:{:02d}m:{:02d}s'.format(d, h, m, s)

    return ft


# 根据配置动态实例化类
def instantiate_from_config(config):
    if config is None:
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")  # 必须有target字段
    module, cls = config["target"].rsplit(".", 1)  # 拆分模块和类名
    #这行代码等价于from module import cls是运行时动态导入的
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", dict()))  # 实例化对象


# 通过字符串获取类
def class_from_string(class_name):
    module, cls = class_name.rsplit(".", 1)  # 拆分模块和类名
    cls = getattr(importlib.import_module(module, package=None), cls)  # 导入类
    return cls


# 获取指定后缀的所有文件
def get_all_file(dir, end_with='.h5'):
    if isinstance(end_with, str):
        end_with = [end_with]
    filenames = []
    for root, dirs, files in os.walk(dir):  # 遍历目录
        for f in files:
            for ew in end_with:
                if f.endswith(ew):  # 匹配后缀
                    filenames.append(os.path.join(root, f))
                    break
    return filenames


# 获取子目录
def get_sub_dirs(dir, abs=True):
    sub_dirs = os.listdir(dir)  # 列出目录
    if abs:
        sub_dirs = [os.path.join(dir, s) for s in sub_dirs]  # 拼接绝对路径
    return sub_dirs


# 获取模型的buffer参数
def get_model_buffer(model):
    state_dict = model.state_dict()  # 获取模型状态字典
    buffers_ = {}
    params_ = {n: p for n, p in model.named_parameters()}  # 获取参数字典

    for k in state_dict:
        if k not in params_:
            buffers_[k] = state_dict[k]  # 只保留buffer
    return buffers_
