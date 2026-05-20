import os  
import torch 
import argparse  
import numpy as np 

from engine.logger import Logger  
from engine.solver import Trainer  
from Data.build_dataloader import build_dataloader, build_dataloader_text
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one  
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config  


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')  
    parser.add_argument('--name', type=str, default=None)  

    parser.add_argument('--config_file', type=str, default=None,
                        help='path of config file') 
    parser.add_argument('--output', type=str, default='OUTPUT',
                        help='directory to save the results') 
    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard for logging') 
    parser.add_argument('--run_multi', type=bool, default=False
                        , help='run multiple times')

    # 随机性相关参数
    parser.add_argument('--cudnn_deterministic', action='store_true', default=True,
                        help='set cudnn.deterministic True') 
    parser.add_argument('--seed', type=int, default=12345,
                        help='seed for initializing training.') 
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                             ' used, and ddp will be disabled')  
    # 训练相关参数
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--mode', type=str, default='infill',
                        help='Infilling or Forecasting.') 
    parser.add_argument('--milestone', type=int, default=10) 

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.') 
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.') 
    parser.add_argument('--cfg_scale', type=float, default=1.5, help='Scale of CFG.') 

    # 修改配置参数
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER) 

    args = parser.parse_args() 
    args.save_dir = os.path.join(args.output, f'{args.name}') 

    return args  



def main():
    args = parse_args()  
    print("CONFIG_FILE:", args.config_file)

    if args.seed is not None:
        seed_everything(args.seed) 

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)  

    config = load_yaml_config(args.config_file) 
    config = merge_opts_to_config(config, args.opts)  

    logger = Logger(args) 
    logger.save_config(config)
    logger.log_info("=====================================================")
    logger.log_info("           ✨ 训练参数配置 (Args & Config) ✨           ")
    logger.log_info("=====================================================")

    
    logger.log_info("\n--- 命令行参数 (Args) ---")
    args_dict = vars(args)
    for key, value in sorted(args_dict.items()):
        logger.log_info(f"  {key.ljust(20)}: {value}")

    logger.log_info("\n--- YAML 配置 (Config) ---")

    try:
        import yaml
        yaml_str = yaml.dump(config, indent=2, default_flow_style=False)
        for line in yaml_str.split('\n'):
            logger.log_info(line)

    except ImportError:
        import pprint
        logger.log_info("（YAML 库未安装，打印配置字典）")
        logger.log_info(pprint.pformat(config))

    logger.log_info("=====================================================")
    logger.log_info("           ✨ 配置打印结束，开始训练... ✨           ")
    logger.log_info("=====================================================\n")
    model = instantiate_from_config(config['model']).cuda()

    if args.train:
        dataloader_info = build_dataloader(config, args)
        trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)
        trainer.train()  
    else:
        text_dataloader_info = build_dataloader_text(config, args)  
        trainer = Trainer(config=config, args=args, model=model, dataloader=text_dataloader_info, logger=logger) 
        trainer.load(args.milestone)  
        dataloader, dataset = text_dataloader_info['dataloader'], text_dataloader_info['dataset']
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        text_emb = torch.tensor(dataset.text_emb, dtype=torch.float32).to(device)
        trend_text_emb = torch.tensor(dataset.trend_emb, dtype=torch.float32).to(device) 
        season_text_emb = torch.tensor(dataset.season_emb, dtype=torch.float32).to(device)  
        print("trainer.cfg_scale =", trainer.cfg_scale)
        print("dataset.condition_dropout_prob =", getattr(dataset, "condition_dropout_prob", None))
        print("text_emb is None =", text_emb is None)
        print("model supports guided_model_predictions =", hasattr(trainer.model, "guided_model_predictions"))
        print("CFG active during sampling =", (trainer.cfg_scale > 1.0) and (text_emb is not None))

        if getattr(args, 'run_multi', True):
            for i in range(10):
                run_dir = os.path.join(args.save_dir, f'run', f'run_{i}')
                os.makedirs(run_dir, exist_ok=True)
                samples = trainer.sample(num=len(dataset), size_every=2001, shape=[dataset.window, dataset.var_num],
                                         text_emb=text_emb,trend_text_emb=trend_text_emb,season_text_emb=season_text_emb)[0]
                # samples = trainer.sample(num=len(dataset), size_every=2001, shape=[dataset.window, dataset.var_num], text_emb=text_emb,trend_text_emb=trend_text_emb,season_text_emb=season_text_emb,resid_text_emb=resid_text_emb, cfg_scale=args.cfg_scale)
                if dataset.auto_norm:
                    samples = unnormalize_to_zero_to_one(samples)
                np.save(os.path.join(run_dir, f'ddpm_fake_{args.name}.npy'), samples)
        else:
            samples = trainer.sample(
                num=len(dataset),
                size_every=2001,
                shape=[dataset.window, dataset.var_num],
                text_emb=text_emb,
                trend_text_emb=trend_text_emb,
                season_text_emb=season_text_emb
            )

            if dataset.auto_norm:
                samples = unnormalize_to_zero_to_one(samples)
                trend_samples = unnormalize_to_zero_to_one(trend_samples)
                season_samples = unnormalize_to_zero_to_one(season_samples)

            np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy'), samples)
            np.save(os.path.join(args.save_dir, f'ddpm_trend_{args.name}.npy'), trend_samples)
            np.save(os.path.join(args.save_dir, f'ddpm_season_{args.name}.npy'), season_samples)

# 程序入口
if __name__ == '__main__':
    main()
