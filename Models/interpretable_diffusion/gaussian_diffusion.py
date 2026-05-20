import math
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from Models.interpretable_diffusion.transformer import Transformer
from Models.interpretable_diffusion.model_utils import default, identity, extract


# 高斯扩散训练器类

def linear_beta_schedule(timesteps):
    # 线性beta调度函数
    scale = 1000 / timesteps  # 缩放因子
    beta_start = scale * 0.0001  # beta起始值
    beta_end = scale * 0.02  # beta结束值
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)  # 生成线性beta序列


def cosine_beta_schedule(timesteps, s=0.008):
    """
    余弦beta调度函数
    参考：https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1  # 步数加1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)  # 生成步数序列
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2  # 计算累计alpha
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # 归一化
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])  # 计算beta
    return torch.clip(betas, 0, 0.999)  # 限制beta范围


class Diffusion_TS(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            beta_schedule='cosine',
            n_heads=4,
            mlp_ratio=4,
            eta=0.0,
            kernel_size=None,
            padding_size=None,
            use_ff=True,
            reg_weight=None,
            **kwargs
    ):
        super(Diffusion_TS, self).__init__()  # 初始化父类

        self.eta, self.use_ff = eta, use_ff  # 采样参数和是否使用傅里叶损失
        self.seq_length = seq_length  # 序列长度
        self.feature_size = feature_size  # 特征维度
        self.ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)  # 傅里叶损失权重

        # 初始化Transformer模型
        self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer=n_layer,
                                 n_heads=n_heads, mlp_ratio=mlp_ratio,
                                 max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size], **kwargs)

        # 选择beta调度方式
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')  # 未知调度报错

        alphas = 1. - betas  # 计算alpha
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # alpha累计连乘
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)  # 前一时刻累计alpha

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)  # 扩散步数
        self.loss_type = loss_type  # 损失类型

        # 采样相关参数
        self.sampling_timesteps = default(
            sampling_timesteps, timesteps)  # 采样步数，默认与训练步数一致

        assert self.sampling_timesteps <= timesteps  # 采样步数不能大于训练步数
        self.fast_sampling = self.sampling_timesteps < timesteps  # 是否快速采样

        # 注册buffer辅助函数，将float64转为float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 扩散过程相关参数
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # 后验分布相关参数
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # 上式等价于 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)
        # 后验方差的对数，防止数值为0
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # 损失加权
        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

    def predict_noise_from_start(self, x_t, t, x0):
        # 根据x_t和预测出来的x0预测噪声
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_start_from_noise(self, x_t, t, noise):
        # 根据x_t和噪声预测x0
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        # 计算后验均值和方差
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def output(self, x, t, text_emb=None, trend_text_emb=None, season_text_emb=None,
               padding_masks=None):
        # Transformer输出，返回趋势和季节分量之和
        model_output = self.model(x, t, text_emb, trend_text_emb, season_text_emb,
                                   padding_masks=padding_masks)
        return model_output

    def model_predictions(self, x, t, text_emb=None, trend_text_emb=None, season_text_emb=None,
                          clip_x_start=False, padding_masks=None,return_decomposition=False):
        # 预测噪声和x0
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = self.output(x, t, text_emb=text_emb, trend_text_emb=trend_text_emb, season_text_emb=season_text_emb,
                             padding_masks=padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def guided_model_predictions(self, x, t, text_emb=None, cfg_scale=1.0, trend_text_emb=None, season_text_emb=None,
                                clip_x_start=False, padding_masks=None,return_decomposition=False):
        # 1. 条件预测
        pred_noise_cond, x_start_cond = self.model_predictions(x, t, text_emb=text_emb, trend_text_emb=trend_text_emb,
                                                               season_text_emb=season_text_emb,
                                                               clip_x_start=clip_x_start,
                                                               padding_masks=padding_masks,return_decomposition=return_decomposition)
        if cfg_scale == 1.0 or text_emb is None:
            return pred_noise_cond, x_start_cond
        # 2. 无条件预测
        # 创建一个全零向量作为无条件嵌入
        uncond_emb = torch.zeros_like(text_emb)
       
        untrend_emb = torch.zeros_like(trend_text_emb)
        unseason_emb = torch.zeros_like(season_text_emb)
        pred_noise_uncond, x_start_uncond = self.model_predictions(x, t, text_emb=uncond_emb,
                                                                   trend_text_emb=untrend_emb,
                                                                   season_text_emb=unseason_emb,
                                                                   clip_x_start=clip_x_start,
                                                                   padding_masks=padding_masks,return_decomposition=return_decomposition)
        # 3. 计算引导噪声 (CFG 公式)
        pred_noise_guided = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)

        # 4. 根据引导噪声重新计算 x_start (用于 clipping)
        x_start_guided = self.predict_start_from_noise(x, t, pred_noise_guided)

        return pred_noise_guided, x_start_guided

    def p_mean_variance(self, x, t, clip_denoised=True, text_emb=None, trend_text_emb=None, season_text_emb=None,
                        cfg_scale=1.0):
        # 计算采样时的均值和方差
        if cfg_scale > 1.0 and text_emb is not None:
            _, x_start = self.guided_model_predictions(x, t, text_emb, cfg_scale, clip_x_start=clip_denoised,
                                                       trend_text_emb=trend_text_emb, season_text_emb=season_text_emb,
                                                      )
        else:
            # 原始逻辑
            _, x_start = self.model_predictions(x, t, text_emb=text_emb, trend_text_emb=trend_text_emb,
                                                season_text_emb=season_text_emb)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def sample(self, shape, text_emb=None, trend_text_emb=None, season_text_emb=None,
               cfg_scale=1.0):
        device = self.betas.device
        img = torch.randn(shape, device=device)

        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            batched_times = torch.full((img.shape[0],), t, device=device)
            if cfg_scale > 1.0 and text_emb is not None:
                _, x_start = self.guided_model_predictions(
                    img, batched_times, text_emb, cfg_scale,
                    clip_x_start=True,
                    trend_text_emb=trend_text_emb,
                    season_text_emb=season_text_emb, return_decomposition=True
                )
            else:
                _, x_start = self.model_predictions(
                    img, batched_times,
                    text_emb=text_emb,
                    trend_text_emb=trend_text_emb,
                    season_text_emb=season_text_emb,
                    clip_x_start=True, return_decomposition=True
                )
            # 更新 img（标准 DDPM 采样）
            model_mean, posterior_variance, posterior_log_variance = \
                self.q_posterior(x_start=x_start, x_t=img, t=batched_times)
            noise = torch.randn_like(img) if t > 0 else 0.
            img = model_mean + (0.5 * posterior_log_variance).exp() * noise

        return img

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True, text_emb=None, trend_text_emb=None, season_text_emb=None,
                    cfg_scale=1.0):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        skip = total_timesteps // sampling_timesteps
        times = list(range(0, total_timesteps, skip))
        if total_timesteps - 1 not in times:
            times.append(total_timesteps - 1)
        if 0 not in times:
            times.append(0)
        times = list(reversed(sorted(times)))[:sampling_timesteps]
        time_pairs = list(zip(times[:-1], times[1:])) + [(times[-1], -1)]

        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            if cfg_scale > 1.0 and text_emb is not None:
                pred_noise, x_start= self.guided_model_predictions(
                    img, time_cond, text_emb, cfg_scale,
                    clip_x_start=clip_denoised,
                    trend_text_emb=trend_text_emb,
                    season_text_emb=season_text_emb, return_decomposition=True
                )
            else:
                pred_noise, x_start= self.model_predictions(
                    img, time_cond,
                    clip_x_start=clip_denoised,
                    text_emb=text_emb,
                    trend_text_emb=trend_text_emb,
                    season_text_emb=season_text_emb, return_decomposition=True
                )

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img

    def generate_mts(self, batch_size=16, text_emb=None, trend_text_emb=None, season_text_emb=None,
                     model_kwargs=None, cond_fn=None, cfg_scale=1.0):
        feature_size, seq_length = self.feature_size, self.seq_length
        if cond_fn is not None:
            raise NotImplementedError("cond_fn not supported with components")
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        fake= sample_fn(
            (batch_size, seq_length, feature_size),
            text_emb=text_emb,
            trend_text_emb=trend_text_emb,
            season_text_emb=season_text_emb,
            cfg_scale=cfg_scale
        )
        return fake 
    @property
    def loss_fn(self):
        # 返回损失函数
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        # 前向扩散采样
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            # x_t = √ᾱₜ * x₀ + √(1 - ᾱₜ) * ε
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(self, x_start, t, target=None, noise=None, padding_masks=None, text_emb=None, trend_text_emb=None,
                    season_text_emb=None):
        # 训练损失计算
        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start

        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # 加噪声
        model_out = self.output(x, t, text_emb=text_emb, trend_text_emb=trend_text_emb, season_text_emb=season_text_emb,
                                 padding_masks=padding_masks)

        train_loss = self.loss_fn(model_out, target, reduction='none')

        fourier_loss = torch.tensor([0.])
        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none') \
                           + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            train_loss += self.ff_weight * fourier_loss

        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    def forward(self, x, text_emb=None, trend_text_emb=None, season_text_emb=None, **kwargs):
        # 前向传播，返回训练损失
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_start=x, t=t, text_emb=text_emb, trend_text_emb=trend_text_emb,
                                season_text_emb=season_text_emb,**kwargs)




if __name__ == '__main__':
    pass
