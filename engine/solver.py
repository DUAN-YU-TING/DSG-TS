import os  # 操作系统相关
import sys  # 系统相关
import time  # 时间相关
import torch  # PyTorch库
import numpy as np  # 数值计算
import torch.nn.functional as F  # PyTorch常用函数

from pathlib import Path  # 路径处理
from tqdm.auto import tqdm  # 进度条
from ema_pytorch import EMA  # 指数滑动平均
from torch.optim import Adam  # Adam优化器
from torch.nn.utils import clip_grad_norm_  # 梯度裁剪
from Utils.io_utils import instantiate_from_config, get_model_parameters_info  # 工具函数

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))  # 添加上级目录到sys.path

# 无限循环数据加载器
def cycle(dl):
    """
    作用：让dataloader无限循环，适用于epoch不定的训练
    """
    while True:
        for data in dl:
            yield data

class Trainer(object):
    """
    训练器类，负责模型训练、保存、加载、采样、恢复、分类器训练等功能
    """
    def __init__(self, config, args, model, dataloader, logger=None):
        """
        初始化Trainer，设置模型、优化器、EMA、调度器等
        """
        super().__init__()
        self.model = model  # 主模型
        self.device = self.model.betas.device  # 设备
        self.train_num_steps = config['solver']['max_epochs']  # 最大训练步数
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']  # 梯度累积步数
        self.save_cycle = config['solver']['save_cycle']  # 保存周期
        self.dl = cycle(dataloader['dataloader'])  # 无限循环的数据加载器
        self.dataloader = dataloader['dataloader']  # 原始数据加载器
        self.step = 0  # 当前步数
        self.milestone = 0  # 当前里程碑
        self.args, self.config = args, config  # 参数和配置
        self.logger = logger  # 日志记录器

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')  # 结果保存路径
        os.makedirs(self.results_folder, exist_ok=True)  # 创建目录

        start_lr = config['solver'].get('base_lr', 1.0e-4)  # 初始学习率
        ema_decay = config['solver']['ema']['decay']  # EMA衰减率
        ema_update_every = config['solver']['ema']['update_interval']  # EMA更新频率

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])  # 优化器
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)  # EMA对象

        sc_cfg = config['solver']['scheduler']  # 调度器配置
        sc_cfg['params']['optimizer'] = self.opt  # 传入优化器
        self.sch = instantiate_from_config(sc_cfg)  # 实例化调度器

        self.cfg_scale = config['solver'].get('cfg_scale', 1.0)  # 默认值设为 1.0

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))  # 打印模型参数信息
        self.log_frequency = 100  # 日志频率

    def save(self, milestone, verbose=False):
        """
        保存当前模型、优化器、EMA等状态到文件
        """
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,  # 当前步数
            'model': self.model.state_dict(),  # 模型参数
            'ema': self.ema.state_dict(),  # EMA参数
            'opt': self.opt.state_dict(),  # 优化器参数
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))  # 保存到文件

    def save_classifier(self, milestone, verbose=False):
        """
        保存分类器状态到文件
        """
        if self.logger is not None and verbose:
            self.logger.log_info('Save current classifer to {}'.format(str(self.results_folder / f'ckpt_classfier-{milestone}.pt')))
        data = {
            'step': self.step_classifier,  # 分类器步数
            'classifier': self.classifier.state_dict()  # 分类器参数
        }
        torch.save(data, str(self.results_folder / f'ckpt_classfier-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        """
        加载模型、优化器、EMA等状态
        """
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)  # 加载文件
        self.model.load_state_dict(data['model'])  # 加载模型参数
        self.step = data['step']  # 恢复步数
        self.opt.load_state_dict(data['opt'])  # 恢复优化器
        self.ema.load_state_dict(data['ema'])  # 恢复EMA
        self.milestone = milestone  # 恢复里程碑

    def load_classifier(self, milestone, verbose=False):
        """
        加载分类器状态
        """
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'ckpt_classfier-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'ckpt_classfier-{milestone}.pt'), map_location=device)
        self.classifier.load_state_dict(data['classifier'])  # 加载分类器参数
        self.step_classifier = data['step']  # 恢复步数
        self.milestone_classifier = milestone  # 恢复里程碑

    def train(self):
        """
        训练主模型
        """
        device = self.device
        step = 0  # 本地步数
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        with tqdm(initial=step, total=self.train_num_steps) as pbar:  # 进度条
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):  # 梯度累积
                    data, text,trend_emb, season_emb= next(self.dl)  # 获取数据
                    data, text,trend_emb, season_emb= data.to(device),text.to(device), trend_emb.to(device), season_emb.to(
                        device)  # 移动到设备
                    loss = self.model(data, text,trend_emb, season_emb)  # 计算损失
                    loss = loss / self.gradient_accumulate_every  # 均分损失
                    loss.backward()  # 反向传播
                    total_loss += loss.item()  # 累加损失

                pbar.set_description(f'loss: {total_loss:.6f}')  # 显示损失

                clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
                self.opt.step()  # 优化器更新
                self.sch.step(total_loss)  # 调度器步进
                self.opt.zero_grad()  # 梯度清零
                self.step += 1  # 全局步数+1
                step += 1  # 本地步数+1
                self.ema.update()  # EMA更新

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:  # 到达保存周期
                        self.milestone += 1
                        self.save(self.milestone)

                    if self.logger is not None and self.step % self.log_frequency == 0:  # 日志记录
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)  # 进度条更新

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def sample(self, num, size_every, shape=None, model_kwargs=None, cond_fn=None, **kwargs):
        """
        采样生成数据，并返回 fake, trend, season 三部分
        """
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')

        # 初始化三个空数组
        num_cycle = int(np.ceil(num / size_every))  # 采样轮数
        text_emb = kwargs.get('text_emb', None)
        trend_text_emb = kwargs.get('trend_text_emb', None)
        season_text_emb = kwargs.get('season_text_emb', None)


        if trend_text_emb is None or season_text_emb is None:
            raise ValueError(
                "Missing one or more required text embeddings (trend_text_emb, season_text_emb) in sample method.")

        fake_list, trend_list, season_list = [], [], []

        for _ in range(num_cycle):
            batch_start = _ * size_every
            batch_end = min(batch_start + size_every, num)
            batch_size = batch_end - batch_start

            batch_text_emb = text_emb[batch_start:batch_end]
            batch_trend_emb = trend_text_emb[batch_start:batch_end]
            batch_season_emb = season_text_emb[batch_start:batch_end]

            # ⚠️ 关键修改：接收三个返回值
            fake, trend, season = self.ema.ema_model.generate_mts(
                batch_size=batch_size,
                text_emb=batch_text_emb,
                trend_text_emb=batch_trend_emb,
                season_text_emb=batch_season_emb,
                cond_fn=cond_fn,
                cfg_scale=self.cfg_scale
            )
            fake_list.append(fake.detach().cpu().numpy())
            trend_list.append(trend.detach().cpu().numpy())
            season_list.append(season.detach().cpu().numpy())

            torch.cuda.empty_cache()

            # 分别拼接
            samples_fake = np.concatenate(fake_list, axis=0)[:num]
            samples_trend = np.concatenate(trend_list, axis=0)[:num]
            samples_season = np.concatenate(season_list, axis=0)[:num]

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))

        # 返回三个分量
        return samples_fake, samples_trend, samples_season

    def forward_sample(self, x_start):
        """
        对输入数据做正向扩散采样，返回加噪后的数据和时间步
        """
        b, c, h = x_start.shape  # 获取形状
        noise = torch.randn_like(x_start, device=self.device)  # 生成噪声
        t = torch.randint(0, self.model.num_timesteps, (b,), device=self.device).long()  # 随机时间步
        x_t = self.model.q_sample(x_start=x_start, t=t, noise=noise).detach()  # 正向扩散
        return x_t, t  # 返回加噪数据和时间步
