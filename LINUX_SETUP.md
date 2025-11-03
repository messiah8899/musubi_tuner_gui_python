# Musubi Tuner Linux 安装指南

本指南将帮助您在Linux系统上安装和运行Musubi Tuner炼丹系统。

## 系统要求

- **操作系统**: Ubuntu 20.04+ / CentOS 8+ / 其他主流Linux发行版
- **Python**: 3.10 - 3.12
- **内存**: 建议16GB以上
- **显卡**: NVIDIA GPU (推荐RTX 3060以上，支持CUDA 12.x)
- **存储**: 至少50GB可用空间

## 安装步骤

### 1. 克隆项目

```bash
git clone <repository-url>
cd musubi-tuner_GUI
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用venv
python3 -m venv venv
source venv/bin/activate

# 或使用conda
conda create -n musubi-tuner python=3.11
conda activate musubi-tuner
```

### 3. 安装依赖

#### 方法1: 使用requirements.txt（推荐）

```bash
# 对于CUDA 12.4用户
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 对于CUDA 12.8用户
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 对于CPU用户
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

#### 方法2: 使用uv（更快）

```bash
# 安装uv
pip install uv

# 安装依赖
uv pip install -r requirements.txt
```

### 4. 设置启动脚本权限

```bash
chmod +x start_training_gui.sh
```

### 5. 启动程序

```bash
./start_training_gui.sh
```

或者直接使用Python：

```bash
python train_gui_new.py
```

## CUDA支持

### 检查CUDA安装

```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查CUDA版本
nvcc --version
```

### 安装CUDA（如果需要）

#### Ubuntu/Debian:
```bash
# 添加NVIDIA包仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# 安装CUDA
sudo apt-get install cuda-toolkit-12-4
```

#### CentOS/RHEL:
```bash
# 添加NVIDIA包仓库
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# 安装CUDA
sudo dnf install cuda-toolkit-12-4
```

## 常见问题解决

### 1. 权限问题

```bash
# 如果遇到权限问题，确保脚本有执行权限
chmod +x start_training_gui.sh

# 如果需要，可以使用sudo运行（不推荐）
sudo ./start_training_gui.sh
```

### 2. Python版本问题

```bash
# 检查Python版本
python3 --version

# 如果版本不对，安装正确版本
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-pip
```

### 3. 依赖安装失败

```bash
# 更新pip
pip install --upgrade pip

# 清理缓存重新安装
pip cache purge
pip install -r requirements.txt --no-cache-dir
```

### 4. CUDA相关错误

```bash
# 检查PyTorch CUDA支持
python -c "import torch; print(torch.cuda.is_available())"

# 重新安装PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 5. 内存不足

```bash
# 增加swap空间
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 永久添加到fstab
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 性能优化

### 1. 环境变量设置

在启动脚本中已经包含了以下优化设置：

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

### 2. GPU内存优化

- 启用低内存模式（在GUI中设置）
- 使用FP8量化（如果支持）
- 调整batch size和网络维度

### 3. 系统优化

```bash
# 设置GPU性能模式
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 300  # 设置功率限制（根据显卡调整）

# 设置CPU性能模式
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## 使用说明

1. **启动程序**: 运行 `./start_training_gui.sh`
2. **访问界面**: 
   - **本地访问**: `http://127.0.0.1:7860/` 或 `http://localhost:7860/`
   - **云服务器访问**: `http://YOUR_SERVER_IP:7860/` (需要确保防火墙开放7860端口)
   - **腾讯云/阿里云等**: 使用服务器的公网IP地址访问
3. **配置模型路径**: 使用Linux路径格式，如 `/home/user/models/` 或相对路径 `./models/`
4. **数据集配置**: 创建TOML配置文件，路径使用正斜杠

### 云服务器注意事项

- 确保安全组/防火墙规则允许7860端口的入站连接
- 如果使用腾讯云轻量应用服务器，需要在控制台开放7860端口
- 程序现在监听 `0.0.0.0:7860`，支持外部访问

## 路径配置示例

```toml
# 数据集配置示例 (dataset.toml)
[general]
resolution = 512
batch_size = 1

[[datasets]]
image_dir = "/home/user/datasets/images"
caption_extension = ".txt"
```

## 故障排除

如果遇到问题，请检查：

1. **日志输出**: 查看终端输出的错误信息
2. **系统资源**: 使用 `htop` 和 `nvidia-smi` 监控资源使用
3. **权限**: 确保对所有文件和目录有适当权限
4. **依赖版本**: 确保所有依赖版本兼容

## 支持

如果遇到Linux特定的问题，请：

1. 检查本文档的常见问题部分
2. 查看项目的GitHub Issues
3. 提供详细的错误日志和系统信息

---

**注意**: 本Linux版本基于原Windows版本改造，保持了所有核心功能的兼容性。