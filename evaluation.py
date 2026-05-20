import sys
import os
import numpy as np
import argparse
import json
import datetime


def calculate_mse(ori_data, gen_data):
    return np.mean((ori_data - gen_data) ** 2)

def calculate_wape(ori_data, gen_data):
    total_absolute_error = np.sum(np.abs(ori_data - gen_data))
    total_actual_value = np.sum(np.abs(ori_data))
    return total_absolute_error / total_actual_value if total_actual_value != 0 else np.nan

def cosine_similarity(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    return np.dot(a_flat, b_flat) / (norm_a * norm_b + 1e-8)

def calculate_mrr_threshold(Y_true, Y_candidates, threshold=0.5):
    """
    Y_true: [batch_size, seq_len, dim]
    Y_candidates: [batch_size, seq_len, dim, num_generations]
    """

    n_batch = Y_true.shape[0]
    n_generations = Y_candidates.shape[3]
    mrr_scores = []

    for batch_idx in range(n_batch):
        similarities = []
        for gen_idx in range(n_generations):
            real_seq = Y_true[batch_idx]
            gen_seq = Y_candidates[batch_idx, :, :, gen_idx]
            sim = cosine_similarity(real_seq, gen_seq)
            similarities.append(sim)

        sorted_indices = np.argsort(similarities)[::-1]
        rank = None
        for idx in sorted_indices:
            if similarities[idx] > threshold:
                rank = idx + 1
                break

        mrr = 1.0 / rank if rank is not None else 0.0
        mrr_scores.append(mrr)

    return np.mean(mrr_scores)

def write_json(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved to {save_path}")

##########################################
#                主函数                  #
##########################################

def main():
    parser = argparse.ArgumentParser(description="集成评估脚本")
    parser.add_argument('--true_path', type=str, required=True, help='真实数据 .npy 文件路径')
    parser.add_argument('--gen_path', type=str, required=True, help='单次生成结果 .npy 文件路径')
    parser.add_argument('--gen_root', type=str, required=True, help='run_multi 生成的目录根路径')
    parser.add_argument('--gen_name', type=str, required=True, help='run_x 下生成结果文件名')
    parser.add_argument('--save_dir', type=str, default='./evaluation_results', help='保存结果目录')
    parser.add_argument('--threshold', type=float, default=0.5, help='MRR 相似度阈值')

    args = parser.parse_args()

    # === 读真实数据 ===
    Y_true = np.load(args.true_path)
    Y_pred = np.load(args.gen_path)

    if Y_true.ndim == 2:
        Y_true = np.expand_dims(Y_true, axis=-1)

    if Y_pred.ndim == 2:
        Y_pred = np.expand_dims(Y_pred, axis=-1)

    # === 检查形状 ===
    if Y_true.shape != Y_pred.shape:
        raise ValueError(f"Shape mismatch: true {Y_true.shape}, pred {Y_pred.shape}")

    # === MSE & WAPE ===
    mse = calculate_mse(Y_true, Y_pred)
    wape = calculate_wape(Y_true, Y_pred)

    #=== 构造 run_multi 结果 ===
    candidate_list = []
    for i in range(10):
        run_dir = os.path.join(args.gen_root, f'run_{i}')
        gen_file = os.path.join(run_dir, args.gen_name)
        if not os.path.isfile(gen_file):
            print(f"Warning: {gen_file} not found. Skip.")
            continue
        Y_candidate = np.load(gen_file)
        if Y_candidate.ndim == 2:
            Y_candidate = np.expand_dims(Y_candidate, axis=-1)
        if Y_true.shape != Y_candidate.shape:
            print(f"Shape mismatch in {run_dir}: true {Y_true.shape}, pred {Y_candidate.shape}. Skip.")
            continue
        candidate_list.append(Y_candidate)
    
    if not candidate_list:
        print("No valid candidate results found for MRR@10.")
        mrr = np.nan
    else:
        # === 堆叠成 [batch, L, D, K] ===
        Y_candidates = np.stack(candidate_list, axis=-1)
        mrr = calculate_mrr_threshold(Y_true, Y_candidates, threshold=args.threshold)


    # === 输出 ===
    result = {
        'MSE': mse,
        'WAPE': wape,
        f'MRR@10_q{args.threshold}': mrr
    }

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(args.save_dir, f"eval_all_{now}.json")
    write_json(result, save_path)


if __name__ == '__main__':
    main()
