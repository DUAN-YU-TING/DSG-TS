import os
import ast
import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class CustomDataset(Dataset):
    def __init__(
            self,
            name,
            feature_size,
            data_root,
            window=64,
            proportion=0.7,
            save2npy=True,
            neg_one_to_one=True,
            seed=123,
            period='train',
            output_dir='./OUTPUT',
            predict_length=None,
            missing_ratio=None,
            style='separate',
            distribution='geometric',
            mean_mask_length=3,
            condition_dropout_prob=0.1
    ):
        super(CustomDataset, self).__init__()
        self.condition_dropout_prob = condition_dropout_prob  # 保存参数
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.csv_data = pd.read_csv(os.path.join(data_root, name + '.csv'),encoding='latin1' )
        self.rawdata, self.scaler, self.text_emb,self.trend_emb, self.season_emb= self.read_data(self.name, self.csv_data)
        # self.rawdata, self.scaler, self.trend_emb, self.season_emb, self.resid_emb = self.read_data(self.name, self.csv_data)
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], feature_size
        self.sample_num_total = max(self.len - self.window + 1, 0)
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.data = self.__normalize(self.rawdata)
        train_data, inference_data,train_text_emb, test_text_emb,train_trend_emb, test_trend_emb, train_season_emb, test_season_emb= self.__getsamples(self.data,self.text_emb, self.trend_emb, self.season_emb, proportion, seed)
        self.text_emb = train_text_emb if period == 'train' else test_text_emb
        self.trend_emb = train_trend_emb if period == 'train' else test_trend_emb
        self.season_emb = train_season_emb if period == 'train' else test_season_emb
        self.samples = train_data if period == 'train' else inference_data
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                self.masking = np.ones(self.samples.shape).astype(bool) 
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, text_emb,trend_emb, season_emb,proportion, seed):
        # 这里直接用，不需要滑动
        assert data.shape[0] == trend_emb.shape[0], f"样本数量不一致！{data.shape[0]} vs {trend_emb.shape[0]}"

        index_save_dir = os.path.join(self.dir, 'samples')
        os.makedirs(index_save_dir, exist_ok=True)
        index_train_file = os.path.join(index_save_dir, f'{self.name}_train_id.npy')
        index_test_file = os.path.join(index_save_dir, f'{self.name}_test_id.npy')

        if self.period == 'train':
            
            train_data, test_data, \
            train_text_emb, test_text_emb, \
            train_trend_emb, test_trend_emb, \
            train_season_emb, test_season_emb, \
            train_id, test_id = self.divide(data, text_emb,trend_emb, season_emb, proportion, seed)
            np.save(index_train_file, train_id)
            np.save(index_test_file, test_id)
        else:
            # 如果是 test，直接读已保存的索引
            assert os.path.exists(index_train_file) and os.path.exists(index_test_file), "索引文件不存在！请先跑一次训练集"
            test_id = np.load(index_test_file)
            train_id = np.load(index_train_file)
            train_data = data[train_id, :]
            test_data = data[test_id, :]
            train_text_emb = text_emb[train_id, :]
            test_text_emb = text_emb[test_id, :]
            train_trend_emb = trend_emb[train_id, :]
            test_trend_emb = trend_emb[test_id, :]
            train_season_emb = season_emb[train_id, :]
            test_season_emb = season_emb[test_id, :]


        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"),
                        self.unnormalize(test_data))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"),
                    self.unnormalize(train_data))

            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"),
                            unnormalize_to_zero_to_one(test_data))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"),
                        unnormalize_to_zero_to_one(train_data))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)

            # 可选保存 embedding
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_text_embedding_test.npy"), test_text_emb)
                np.save(os.path.join(self.dir, f"{self.name}_trend_embedding_test.npy"), test_trend_emb)
                np.save(os.path.join(self.dir, f"{self.name}_season_embedding_test.npy"), test_season_emb)
            np.save(os.path.join(self.dir, f"{self.name}_text_embedding_train.npy"), train_text_emb)
            np.save(os.path.join(self.dir, f"{self.name}_trend_embedding_train.npy"), train_trend_emb)
            np.save(os.path.join(self.dir, f"{self.name}_season_embedding_train.npy"), train_season_emb)

        return train_data, test_data, \
            train_text_emb, test_text_emb, \
            train_trend_emb, test_trend_emb, \
            train_season_emb, test_season_emb

    def normalize(self, sq):
        d = self.scaler.transform(sq)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        if sq.ndim == 3:  
            sq = sq.squeeze(-1)
        d = self.__unnormalize(sq.reshape(-1, self.window))
        return d.reshape(-1, self.window, self.var_num)

    def __normalize(self, rawdata):
        if rawdata.ndim == 3:
            rawdata = rawdata.squeeze(-1)
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)

    @staticmethod
    def divide(data, text_emb,trend_emb, season_emb,ratio, seed=2023):
        assert data.shape[0] == text_emb.shape[0], "data 与 trend_emb 行数必须一致"
        assert data.shape[0] == trend_emb.shape[0], "data 与 trend_emb 行数必须一致"
        assert data.shape[0] == season_emb.shape[0], "data 与 season_emb 行数必须一致"
        size = data.shape[0]
        st0 = np.random.get_state()
        np.random.seed(seed)

        id_rdm = np.random.permutation(size)
        split = int(np.ceil(size * ratio))
        train_id = id_rdm[:split]
        test_id = id_rdm[split:]

        train_data = data[train_id, :]
        test_data = data[test_id, :]
        train_text_emb = text_emb[train_id, :]
        test_text_emb = text_emb[test_id, :]
        train_trend_emb = trend_emb[train_id, :]
        test_trend_emb = trend_emb[test_id, :]
        train_season_emb = season_emb[train_id, :]
        test_season_emb = season_emb[test_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return train_data, test_data, \
            train_text_emb, test_text_emb, \
            train_trend_emb, test_trend_emb, \
            train_season_emb, test_season_emb, \
            train_id, test_id

    @staticmethod
    def read_data(name, json_root_data):
        """Reads a single .csv
        """
        # 时间序列处理（对OT列做处理）
        parsed_data = [ast.literal_eval(item) if isinstance(item, str) else item for item in json_root_data['OT']]

        time_series = np.array(parsed_data)

        scaler = MinMaxScaler()
        scaler = scaler.fit(time_series)

        # 这三个不一样
        text_emb = np.array([
            [float(num) for num in x.replace('[', '').replace(']', '').replace(',', '').strip().split()]
            for x in json_root_data['TextEmbedding']
        ])
        # 趋势 embedding
        trend_emb = np.array([
            [float(num) for num in x.replace('[', '').replace(']', '').replace(',', '').strip().split()]
            for x in json_root_data['TrendTextEmb']
        ])

        # 季节 embedding
        season_emb = np.array([
            [float(num) for num in x.replace('[', '').replace(']', '').replace(',', '').strip().split()]
            for x in json_root_data['SeasonTextEmb']
        ])

        return time_series, scaler,text_emb,trend_emb, season_emb


    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  
            masks[idx, :, :] = mask 

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        np.random.set_state(st0)

        return masks.astype(bool)

    def __getitem__(self, ind):
        x = self.samples[ind] 

        text_emb = self.text_emb[ind] 
        trend_emb = self.trend_emb[ind] 
        season_emb = self.season_emb[ind]
        if self.period == 'train' and self.condition_dropout_prob > 0:
            # 检查是否应该随机丢弃
            if np.random.rand() < self.condition_dropout_prob:
                text_emb = np.zeros_like(text_emb)
                trend_emb = np.zeros_like(trend_emb)
                season_emb = np.zeros_like(season_emb)
        x = torch.tensor(x, dtype=torch.float32)

        # 保证 shape: [C, T]
        if x.ndim == 1:
            x = x.unsqueeze(0) 
        elif x.ndim == 2:
            x = x.permute(1, 0) 

        return (
            x.transpose(0, 1),  # (T, C)
            # 这三个不一样
            torch.tensor(text_emb, dtype=torch.float32),
            torch.tensor(trend_emb, dtype=torch.float32),
            torch.tensor(season_emb, dtype=torch.float32),
        )

    def __len__(self):
        return self.sample_num



