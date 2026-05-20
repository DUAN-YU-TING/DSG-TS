import torch  # 导入PyTorch主库
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange, reduce, repeat
from Models.interpretable_diffusion.model_utils import LearnablePositionalEncoding, Conv_MLP, Transpose,AdaLayerNorm
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp 

class FullAttention(nn.Module):
    def __init__(self,
                 n_embd, 
                 n_head, 
                 attn_pdrop=0.1, 
                 resid_pdrop=0.1,
    ):
        super().__init__()
        assert n_embd % n_head == 0

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)


        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) 
        att = self.attn_drop(att)
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        att = att.mean(dim=1, keepdim=False)

        y = self.resid_drop(self.proj(y))
        return y, att

class TextTimeCrossAttention(nn.Module):
    def __init__(self, n_embd, n_head, seq_len):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(n_embd, num_heads=n_head, bias=True)
        self.text_to_time = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, seq_len * n_embd)
        )
        self.norm = nn.LayerNorm(n_embd)

        self.gate = nn.Sequential(
                nn.Linear(n_embd * 2, n_embd),
                nn.ReLU(),
                nn.Linear(n_embd, 1),
                nn.Sigmoid()
            )

    def forward(self, x, text_input):

        B, T, D = x.shape
        shifted_x = self.norm(x)
        query = shifted_x.transpose(0, 1)  # (T, B, D)

        if text_input is None:
            cross_attn_out = torch.zeros_like(x)
            gate_msa = torch.zeros(B, 1, device=x.device)
        else:
            text_time = self.text_to_time(text_input).view(B, T, D)
            key = text_time.transpose(0, 1)
            value = key
            cross_attn_out, _ = self.cross_attn(query, key, value)
            cross_attn_out = cross_attn_out.transpose(0, 1)  # (B, T, D)

            gate_msa = self.gate(torch.cat([x.mean(dim=1), text_input], dim=-1))

        # 残差融合
        x = x + gate_msa.unsqueeze(1) * cross_attn_out
        return x

class AdaLN_DiTBlock(nn.Module):
    def __init__(self, n_embd, n_heads=4, mlp_ratio=4.0,attn_pdrop=0.0,
                 resid_pdrop=0.1):
        super().__init__()
        act_class = nn.GELU
        # 生成调制参数: [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),  # SiLU激活
            nn.Linear(n_embd, 6 * n_embd, bias=True)  # 线性层
        )
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        self.attn = Attention(
            n_embd,
            num_heads=n_heads,
            qkv_bias=True,
            attn_drop=attn_pdrop,
            proj_drop=attn_pdrop,
        )
        mlp_hidden_dim = int(n_embd * mlp_ratio)  # MLP隐藏层维度
        self.mlp = Mlp(in_features=n_embd, hidden_features=mlp_hidden_dim, act_layer=act_class, drop=resid_pdrop)  # MLP
    def forward(self, x, c):
        """
        x: (B, T, D)
        c: (B, text_dim) 文本全局特征向量
        """
        # 生成 AdaLN 调制参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # 注意力部分
        shifted_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(shifted_x)  # 输入需要是 [B, T, C]
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP 部分
        shifted = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(shifted)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x

class TrendBlock(nn.Module):
    """
    Model trend of time series using the polynomial regressor.
    """
    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super(TrendBlock, self).__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )

        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, x):
        b, c, h = x.shape

        x = self.trend(x)
        trend_vals = torch.matmul(x, self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals

class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple

def modulate(x, shift, scale):
    """
    对输入x进行缩放和平移，用于自适应归一化。
    典型的AdaLN（Adaptive LayerNorm）调制
    用条件向量（比如时间步嵌入 t 和文本嵌入 text_input）生成 shift 和 scale 参数
    对每一层 Transformer 的 LayerNorm 后的输出进行调制。
    参数:
        x: 输入张量
        shift: 平移参数
        scale: 缩放参数
    返回:
        调制后的张量
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # 广播实现调制

class TimeEmbedding(nn.Module):
    """
    时间步嵌入模块，将时间步t编码为高维向量。
    参数:
        dim: 嵌入维度，必须为偶数。
    """
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()  # 初始化父类
        self.dim = dim  # 嵌入维度
        assert dim % 2 == 0, "Dimension must be even"  # 必须为偶数

    def forward(self, t):
        """
        前向传播，将时间步t编码为sin/cos嵌入。
        参数:
            t: (B,) 形状的时间步张量
        返回:
            (B, dim) 形状的嵌入张量
        """
        t = t.float().unsqueeze(-1) * 100.0  # 放大时间步
        freqs = torch.pow(10000, torch.linspace(0, 1, self.dim // 2)).to(t.device)  # 频率
        sin_emb = torch.sin(t / freqs)  # 正弦嵌入
        cos_emb = torch.cos(t / freqs)  # 余弦嵌入
        embedding = torch.cat([sin_emb, cos_emb], dim=-1)  # 拼接
        return embedding

class Transformerblock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_channel,
                 n_feat,
                 n_embd=1024,
                 n_heads=4,
                 mlp_ratio=2.0,
                 time_emb_dim=64,
                 attn_pdrop=0.0,
                 resid_pdrop=0.1,
                 activate='GELU'
                 ):
        super().__init__()
        mlp_hidden_dim = int(n_embd * mlp_ratio)  # MLP隐藏层维度
        assert activate in ['GELU', 'GELU2']
        act_class = nn.GELU
        act = nn.GELU()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln1_1 = AdaLayerNorm(n_embd)
        self.ln1_2 = AdaLayerNorm(n_embd)
        self.ln1_3 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn1 = FullAttention(
            n_embd=n_embd,
            n_head=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        self.attn2 = FullAttention(
            n_embd=n_embd,
            n_head=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        self.attn3 = TextTimeCrossAttention(
            n_embd=n_embd,
            n_head=n_heads,
            seq_len=n_channel,
        )
        self.attn4 = TextTimeCrossAttention(
            n_embd=n_embd,
            n_head=n_heads,
            seq_len=n_channel,
        )
        self.mlp = Mlp(in_features=n_embd, hidden_features=mlp_hidden_dim, act_layer=act_class, drop=resid_pdrop)  # MLP
        self.cross_attn = nn.MultiheadAttention(n_embd, num_heads=n_heads, bias=True)

        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat, act=act) 

        self.seasonal = FourierLayer(d_model=n_embd) 
        self.proj = nn.Conv1d(n_channel, n_channel * 2, kernel_size=1)
        self.linear = nn.Linear(n_embd, n_feat)  
    def forward(self, x, c, trend_text_input, season_text_input):
        B, T, D = x.shape  # [Batch, TimeSteps, Channels]

        a, att = self.attn1(self.ln1(x, c))
        x = x + a

        x_r = x
        x1, x2 = self.proj(x_r).chunk(2, dim=1) 

        x1 = self.attn3(self.ln1_1(x1, c), trend_text_input)
        x2 = self.attn4(self.ln1_2(x2, c), season_text_input)
        trend, season = self.trend(x1), self.seasonal(x2)  

        a, att = self.attn1(self.ln1(x, c))
        x = x + a

        m = torch.mean(x, dim=1, keepdim=True) 
        return x - m, self.linear(m), trend, season

class Transformerlayer(nn.Module):
    def __init__(
        self,
            n_channel,
            n_feat,
            n_embd,
            n_heads=4,
            n_layer=6,
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            block_activate='GELU',
            mlp_ratio=2.0,
    ):
        super().__init__()
        self.d_model = n_embd 
        self.n_feat = n_feat

        self.blocks = nn.Sequential(*[Transformerblock(
            n_channel=n_channel, 
            n_feat=n_feat,
            n_embd=n_embd,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            activate=block_activate,
            mlp_ratio=mlp_ratio,
        ) for _ in range(n_layer)])

    def forward(self, x, t, trend_text_input=None, season_text_input=None,padding_masks=None):
        b, c, _ = x.shape  
        mean = []
        season = torch.zeros((b, c, self.d_model), device=x.device) 
        trend = torch.zeros((b, c, self.n_feat), device=x.device) 
        for block in self.blocks:
            x, residual_mean, residual_trend, residual_season= block(x, t,trend_text_input,season_text_input)

            season += residual_season 
            trend += residual_trend 
            mean.append(residual_mean) 

        mean = torch.cat(mean, dim=1)  
        return x, mean, trend, season 


class Transformer(nn.Module):
    def __init__(
        self,
        n_feat,
        n_channel,
        n_layer=6,
        n_embd=1024,
        n_heads=4,
        text_emb_dim=128,  
        time_emb_dim=64, 
        max_len=2048, 
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        block_activate='GELU',
        conv_params=None,
        mlp_ratio=2.0,
        gate_type='dimwise'
    ):
        super().__init__()
        self.emb1 = Conv_MLP(n_feat, n_embd)
        self.inverse1 = Conv_MLP(n_embd, n_feat)
        self.inverse2 = Conv_MLP(n_embd, n_feat)

        self.time_proj = nn.Linear(time_emb_dim, n_embd)
        self.time_embed = TimeEmbedding(time_emb_dim)

        self.text_proj = nn.Linear(text_emb_dim, n_embd)
        self.trend_proj = nn.Linear(text_emb_dim, n_embd)
        self.season_proj = nn.Linear(text_emb_dim, n_embd)

        self.trend = nn.Linear(n_feat, n_embd)
        self.mean = nn.Linear(n_feat, n_embd)

        if conv_params is None or conv_params[0] is None:
            if n_feat < 32 and n_channel < 64:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 5, 2
        else:
            kernel_size, padding = conv_params

        self.combine_s = nn.Conv1d(n_embd, n_feat, kernel_size=kernel_size, stride=1, padding=padding,
                                   padding_mode='circular', bias=False)
        self.combine_m1 = nn.Conv1d(n_layer, 1, kernel_size=1, stride=1, padding=0,
                                   padding_mode='circular', bias=False)
        self.combine_m2 = nn.Conv1d(n_layer, n_channel, kernel_size=1, stride=1, padding=0,
                                   padding_mode='circular', bias=False)
        self.pos_enc1 = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

        self.decoder = Transformerlayer(
            n_channel=n_channel,
            n_feat=n_feat,
            n_embd=n_embd,
            n_heads=n_heads,
            n_layer=n_layer,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            block_activate=block_activate,
            mlp_ratio=mlp_ratio,
        )
        self.text_dit_layers = nn.ModuleList([
            AdaLN_DiTBlock(n_embd=n_embd,n_heads=n_heads)
            for _ in range(3)  # 可以调节层数
        ])
    def forward(self, input, t, text_input,trend_text_input,season_text_input,padding_masks=None):

        emb = self.emb1(input)
        inp_enc = self.pos_enc1(emb)
        trend_text_input = self.trend_proj(trend_text_input)
        season_text_input = self.season_proj(season_text_input)


        output, mean, trend, season = self.decoder(inp_enc, t,
                                                     trend_text_input=trend_text_input,
                                                     season_text_input=season_text_input,
                                                     padding_masks=padding_masks)

        trend = self.trend(trend)
        mean = self.mean(self.combine_m2(mean))
        x_total=trend+season+output+mean
        t_emb = self.time_embed(t) 
        t_cond = self.time_proj(t_emb)  

        if text_input is not None:
              text_condition= self.text_proj(text_input)  
              c = t_cond + text_condition     
        else:
              dummy_text = torch.zeros((input.size(0), self.text_proj.in_features),
                                       device=input.device)
              text_condition = self.text_proj(dummy_text)
              c = t_cond
        X_ts=x_total
        for layer in self.text_dit_layers:
            X_ts = layer(X_ts, c)

        X_ts=self.inverse2(X_ts)

        return X_ts