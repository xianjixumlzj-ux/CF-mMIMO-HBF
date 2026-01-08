import numpy as np
from scipy.io import loadmat
import os

# ===================== 系统参数与路径 =====================
Us = 4
Mr = 64  # 每个AP的天线数
K = 32  # 目标用户数
sample_num = 135  # 目标样本数
noise_power_dbm = 130
dataset_save_path = "dataSet4x64x8x4/130dB/dataSet_130.npy"
params_file = "D:\\py_for_paper\\CF-mMIMO-HBF-main\\O1_28\\O1_28.params.mat"
o1_28_folder = "D:\\py_for_paper\\CF-mMIMO-HBF-main\\O1_28"


# ===================== 1. 加载参数文件 =====================
params_data = loadmat(params_file)
num_BS = int(params_data["num_BS"])
print(f"基站总数：{num_BS}")
active_bs_ids = [4, 5, 8, 9]
for bs_id in active_bs_ids:
    if bs_id < 1 or bs_id > num_BS:
        print(f"错误：基站ID={bs_id}超出范围（1~{num_BS}）")
        exit(1)


# ===================== 2. 读取CIR文件并处理 =====================
bs_channels = []
for bs_id in active_bs_ids:
    cir_file = f"O1_28.{bs_id}.CIR.mat"
    file_path = os.path.join(o1_28_folder, cir_file)
    print(f"加载基站{bs_id}的CIR文件：{file_path}")
    mat_data = loadmat(file_path)
    cir_raw = mat_data["CIR_array_full"]
    cir_length = cir_raw.size
    print(f"CIR长度：{cir_length}")

    if np.iscomplexobj(cir_raw):
        cir_complex = cir_raw
    elif cir_raw.ndim >= 3 and cir_raw.shape[-1] == 2:
        cir_complex = cir_raw[..., 0] + 1j * cir_raw[..., 1]
    else:
        cir_flat = cir_raw.flatten()
        # 修正为偶数长度并转为复数
        if cir_flat.size % 2 != 0:
            cir_flat = cir_flat[:-1]
            print(f"修正为偶数长度：{cir_flat.size}")
        cir_complex = cir_flat[::2] + 1j * cir_flat[1::2]
    print(f"转为复数后形状：{cir_complex.shape}")

    # 重塑为（样本数, 用户数, 天线数）
    if cir_complex.ndim >= 3:
        dims = list(cir_complex.shape)
        if Mr not in dims:
            raise ValueError(f"CIR维度中未找到Mr={Mr}，请检查CIR_array_full结构：{dims}")
        mr_axis = dims.index(Mr)
        remaining_axes = [i for i in range(len(dims)) if i != mr_axis]
        if len(remaining_axes) < 2:
            raise ValueError(f"CIR维度不足以解析(样本数, 用户数, 天线数)：{dims}")
        sample_axis, user_axis = remaining_axes[:2]
        if dims[user_axis] < K:
            raise ValueError(f"CIR中用户数不足：K_available={dims[user_axis]}, K={K}")
        cir_3d = np.moveaxis(cir_complex, [sample_axis, user_axis, mr_axis], [0, 1, 2])
        target_samples = min(sample_num, cir_3d.shape[0])
        cir_cropped = cir_3d[:target_samples, :K, :Mr]
    else:
        max_samples = cir_complex.size // (K * Mr)
        target_samples = min(sample_num, max_samples)
        if target_samples == 0:
            raise ValueError(f"CIR长度不足以支持K/Mr设置：len={cir_complex.size}, needed={K * Mr}")
        needed = target_samples * K * Mr
        cir_3d = cir_complex[:needed].reshape(target_samples, K, Mr)
        cir_cropped = cir_3d

    print(f"重塑为：（样本数={cir_cropped.shape[0]}, 用户数={cir_cropped.shape[1]}, 天线数={cir_cropped.shape[2]}）")
    # 扩展为（样本数, 1, 用户数, 天线数），方便后续合并
    bs_channels.append(cir_cropped[:, np.newaxis, :, :])


# ===================== 3. 合并信道并生成数据集 =====================
# 合并后形状：（样本数, 基站数, 用户数, 天线数）
channel_matrix = np.concatenate(bs_channels, axis=1)
sample_num_actual = channel_matrix.shape[0]
N_BS_actual = channel_matrix.shape[1]
K_actual = channel_matrix.shape[2]
Mr_actual = channel_matrix.shape[3]
print(f"最终信道矩阵形状：{channel_matrix.shape}（样本数, 基站数, 用户数, 天线数）")
if K_actual != K:
    raise ValueError(f"K_actual与目标K不一致：{K_actual} vs {K}")
if Mr_actual != Mr:
    raise ValueError(f"Mr_actual与目标Mr不一致：{Mr_actual} vs {Mr}")

# 筛选服务用户（每个样本取前Us个强用户）
# 计算每个用户的平均功率（沿基站和天线维度）
channel_power = np.abs(channel_matrix) **2
avg_power_per_ue = np.mean(channel_power, axis=(1, 3))  # （样本数, 用户数）
top_us_ue_indices = np.argsort(avg_power_per_ue, axis=1)[:, -Us:]  # （样本数, Us）

# 初始化服务用户信道（形状：样本数, 基站数, 天线数, Us）
service_ue_channel = np.zeros((sample_num_actual, N_BS_actual, Mr_actual, Us), dtype=np.complex64)
for i in range(sample_num_actual):
    # 提取当前样本的服务用户信道，形状：（基站数, Us, 天线数）
    extracted = channel_matrix[i, :, top_us_ue_indices[i], :]
    # 调整维度顺序为（基站数, 天线数, Us），匹配目标形状
    extracted = extracted.transpose(0, 2, 1)  # 交换后两个维度
    service_ue_channel[i] = extracted  # 此时形状完全匹配


# 展平信道特征（顺序需匹配工程读取的[Us, Mr, N_BS]）
channel_flat = service_ue_channel.transpose(0, 3, 2, 1).reshape(sample_num_actual, -1)

# 计算RSSI特征
service_power = np.abs(service_ue_channel)** 2
avg_service_power = np.mean(service_power, axis=2)  # （样本数, 基站数, Us）
rssi_matrix = np.zeros((sample_num_actual, N_BS_actual, Us, K), dtype=np.float32)
noise_power = 10 **(-noise_power_dbm / 10)

for i in range(sample_num_actual):
    rssi_matrix[i, :, :, top_us_ue_indices[i]] = avg_service_power[i, :, :, np.newaxis]
    mask = ~np.isin(np.arange(K), top_us_ue_indices[i])
    rssi_matrix[i, :, :, mask] = noise_power

rssi_dbm = 10 * np.log10(rssi_matrix + 1e-12)
rssi_norm = (rssi_dbm - np.mean(rssi_dbm)) / (np.std(rssi_dbm) + 1e-8)
rssi_flat = rssi_norm.transpose(0, 2, 1, 3).reshape(sample_num_actual, -1)

# 保存最终数据集
dataset = np.concatenate([channel_flat, rssi_flat], axis=1)
os.makedirs(os.path.dirname(dataset_save_path), exist_ok=True)
np.save(dataset_save_path, dataset)

print(f"数据集生成成功！形状：{dataset.shape}")

dataset_dir = os.path.dirname(dataset_save_path)
dataset_md_path = os.path.join(dataset_dir, "DATASET.md")
dataset_md_lines = [
    f"Us = {Us}\n",
    f"Mr = {Mr}\n",
    "Nrf = 8\n",
    f"N_BS = {N_BS_actual}\n",
    f"K = {K}\n",
    "Reserved = 0\n",
    "Reserved2 = 0\n",
    f"Noise_pwr = {noise_power_dbm}\n",
]
with open(dataset_md_path, "w", encoding="utf-8") as f:
    f.writelines(dataset_md_lines)

dataset = np.load(dataset_save_path)
print("数据集形状：", dataset.shape)
channel_len = Us * Mr * N_BS_actual
rssi_len = Us * N_BS_actual * K
print("信道长度：", dataset.shape[1] - rssi_len)
print("RSSI长度：", rssi_len)
