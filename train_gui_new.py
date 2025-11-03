import gradio as gr
import subprocess
import sys
import os
import signal
import psutil
from typing import Generator
import toml
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import re
import threading
import time
from collections import deque
from matplotlib import font_manager
import tempfile


def _setup_matplotlib_cjk_font():
    try:
        available_fonts = {f.name for f in font_manager.fontManager.ttflist}
        candidates = [
            'Microsoft YaHei',
            'Microsoft YaHei UI',
            'SimHei',
            'Noto Sans CJK SC',
            'Source Han Sans CN'
        ]
        chosen = None
        for name in candidates:
            if name in available_fonts:
                chosen = name
                break
        if chosen:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [chosen]
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

_setup_matplotlib_cjk_font()


running_processes = {
    "cache": None,
    "train": None
}

# 全局变量存储训练数据
training_data = {
    'steps': deque(maxlen=1000),
    'losses': deque(maxlen=1000),
    'timestamps': deque(maxlen=1000),
    'learning_rates': deque(maxlen=1000),
    'epochs': deque(maxlen=1000)
}

def parse_training_log(line):
    """解析训练日志中的loss和其他指标"""
    # 匹配多种格式的训练日志
    patterns = [
        # 格式1: steps: 0% | 20/10576 [00:32<4:23:26, 1.50s/it, avg_loss=0.113] 或 avr_loss=0.113
        r'steps:\s*(\d+)%\s*\|\s*(\d+)/(\d+)\s*\[.*?,\s*av[gr]_loss=([0-9.]+)\]',
        # 格式2: | 10/10576 [00:18<5:32:12, 1.89s/it, avg_loss=0.112] 或 avr_loss=0.112
        r'\|\s*(\d+)/(\d+)\s*\[.*?,\s*av[gr]_loss=([0-9.]+)\]',
        # 格式3: steps: 0%|10/10576[...]（无空格也支持）
        r'steps:\s*(\d+)%\s*\|\s*(\d+)/(\d+)\s*\[.*?,\s*av[gr]_loss=([0-9.]+)\]'
    ]

    for i, pattern in enumerate(patterns):
        match = re.search(pattern, line)
        if match:
            if i == 0 or i == 2:  # 有百分比的格式
                progress_percent = int(match.group(1))
                current_step = int(match.group(2))
                total_steps = int(match.group(3))
                avg_loss = float(match.group(4))
            else:  # 没有百分比的格式
                current_step = int(match.group(1))
                total_steps = int(match.group(2))
                avg_loss = float(match.group(3))
                progress_percent = int((current_step / total_steps) * 100) if total_steps > 0 else 0

            # 计算当前epoch（估算）
            current_epoch = (current_step / total_steps) * 100 if total_steps > 0 else 0

            # 添加到全局数据
            training_data['steps'].append(current_step)
            training_data['losses'].append(avg_loss)
            training_data['timestamps'].append(datetime.now())
            training_data['epochs'].append(current_epoch)
            training_data['learning_rates'].append(1e-5)  # 默认学习率，可以从参数中获取



            return {
                'step': current_step,
                'total_steps': total_steps,
                'loss': avg_loss,
                'progress': progress_percent,
                'epoch': current_epoch
            }
    return None

project_root = Path(__file__).parent.absolute()
src_path = project_root / "src"

def create_loss_plot():
    """创建Loss曲线图"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    if len(training_data['steps']) > 0:
        steps = list(training_data['steps'])
        losses = list(training_data['losses'])

        # 绘制loss曲线
        ax.plot(steps, losses, color='#64b5f6', linewidth=2, alpha=0.9, label='Average Loss', marker='o', markersize=3)

        # 添加趋势线
        if len(steps) > 5:
            import numpy as np
            z = np.polyfit(steps, losses, 1)
            p = np.poly1d(z)
            ax.plot(steps, p(steps), color='#ff7043', linestyle='--', alpha=0.7, label='Trend', linewidth=1.5)

        # 显示最新的loss值
        if len(losses) > 0:
            latest_loss = losses[-1]
            latest_step = steps[-1]
            ax.annotate(f'Latest: {latest_loss:.4f}',
                       xy=(latest_step, latest_loss),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#ff7043', alpha=0.7),
                       color='white', fontsize=10, fontweight='bold')

        # 设置图表样式
        ax.set_xlabel('Training Steps', color='white', fontsize=11)
        ax.set_ylabel('Loss', color='white', fontsize=11)
        ax.set_title('实时 Loss 曲线', color='white', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, color='gray', linestyle=':')
        ax.legend(facecolor='#16213e', edgecolor='white', labelcolor='white', fontsize=10)

        # 设置坐标轴颜色
        ax.tick_params(colors='white', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('white')
    else:
        ax.text(0.5, 0.5, '等待训练数据...',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='white', fontsize=14)
        ax.set_title('实时 Loss 曲线', color='white', fontsize=13, fontweight='bold')
        ax.set_xlabel('Training Steps', color='white', fontsize=11)
        ax.set_ylabel('Loss', color='white', fontsize=11)
        for spine in ax.spines.values():
            spine.set_color('white')

    plt.tight_layout()
    return fig

def create_progress_plot():
    """创建训练进度和统计信息图"""
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax1.set_facecolor('#16213e')
    ax2.set_facecolor('#16213e')

    if len(training_data['steps']) > 0:
        steps = list(training_data['steps'])
        epochs = list(training_data['epochs'])
        losses = list(training_data['losses'])

        # 上图：训练进度百分比
        progress_percent = [(step / max(steps)) * 100 if max(steps) > 0 else 0 for step in steps]
        ax1.plot(steps, progress_percent, color='#4caf50', linewidth=2, alpha=0.9, label='训练进度', marker='s', markersize=2)
        ax1.set_ylabel('进度 (%)', color='white', fontsize=11)
        ax1.set_title('训练进度 & Loss 统计', color='white', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, color='gray', linestyle=':')
        ax1.legend(facecolor='#16213e', edgecolor='white', labelcolor='white', fontsize=10)
        ax1.tick_params(colors='white', labelsize=9)

        # 显示当前进度
        if len(progress_percent) > 0:
            current_progress = progress_percent[-1]
            ax1.annotate(f'{current_progress:.1f}%',
                        xy=(steps[-1], current_progress),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#4caf50', alpha=0.7),
                        color='white', fontsize=9, fontweight='bold')

        # 下图：Loss变化率
        if len(losses) > 1:
            loss_changes = [0] + [losses[i] - losses[i-1] for i in range(1, len(losses))]
            colors = ['#ff5722' if change > 0 else '#2196f3' for change in loss_changes]
            ax2.bar(steps, loss_changes, color=colors, alpha=0.7, width=max(steps)*0.01 if max(steps) > 0 else 1)
            ax2.axhline(y=0, color='white', linestyle='-', alpha=0.5, linewidth=1)

        ax2.set_xlabel('Training Steps', color='white', fontsize=11)
        ax2.set_ylabel('Loss 变化', color='white', fontsize=11)
        ax2.grid(True, alpha=0.3, color='gray', linestyle=':')
        ax2.tick_params(colors='white', labelsize=9)

        # 设置坐标轴颜色
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_color('white')
    else:
        ax1.text(0.5, 0.5, '等待训练数据...',
                horizontalalignment='center', verticalalignment='center',
                transform=ax1.transAxes, color='white', fontsize=14)
        ax1.set_title('训练进度 & Loss 统计', color='white', fontsize=13, fontweight='bold')

        ax2.text(0.5, 0.5, '等待训练数据...',
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, color='white', fontsize=14)

        # 设置坐标轴颜色
        for ax in [ax1, ax2]:
            ax.set_xlabel('Training Steps', color='white', fontsize=11)
            ax.set_ylabel('', color='white', fontsize=11)
            for spine in ax.spines.values():
                spine.set_color('white')

    plt.tight_layout()
    return fig

def get_env_with_pythonpath():
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{current_pythonpath}"
    else:
        env["PYTHONPATH"] = str(src_path)
    return env

def terminate_process_tree(proc: subprocess.Popen):
    if proc is None:
        return
    try:
        parent_pid = proc.pid
        if parent_pid is None:
            return
        parent = psutil.Process(parent_pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        print(f"[WARN] terminate_process_tree error: {e}")

def stop_caching():
    if running_processes["cache"] is not None:
        proc = running_processes["cache"]
        if proc.poll() is None:
            terminate_process_tree(proc)
            running_processes["cache"] = None
            return "[INFO] 缓存进程已停止\n"
        else:
            return "[WARN] 缓存进程已完成\n"
    else:
        return "[WARN] 没有运行中的缓存进程\n"

def stop_training():
    if running_processes["train"] is not None:
        proc = running_processes["train"]
        if proc.poll() is None:
            terminate_process_tree(proc)
            running_processes["train"] = None
            return "[INFO] 训练进程已停止\n"
        else:
            return "[WARN] 训练进程已完成\n"
    else:
        return "[WARN] 没有运行中的训练进程\n"

SETTINGS_FILE = "settings.toml"

def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = toml.load(f)
                return settings
        except Exception:
            return {}
    else:
        return {}

def save_settings(settings: dict):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            toml.dump(settings, f)
    except Exception as e:
        print(f"[WARN] Failed to save settings.toml: {e}")

def get_dataset_config(file_path: str, text_path: str) -> str:
    if file_path and os.path.isfile(file_path):
        return file_path
    elif text_path.strip():
        return text_path.strip()
    else:
        return ""

import platform
import shutil

def get_python_executable():
    """获取跨平台的Python可执行文件路径"""
    # 首先检查是否在虚拟环境中
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # 在虚拟环境中，使用当前Python
        return sys.executable
    
    # 检查是否有嵌入式Python（Windows）
    if platform.system() == "Windows":
        embedded_python = "./python_embeded/python.exe"
        if os.path.exists(embedded_python):
            return embedded_python
    
    # 尝试查找系统Python
    python_candidates = ["python3", "python"]
    for candidate in python_candidates:
        python_path = shutil.which(candidate)
        if python_path:
            return python_path
    
    # 最后使用当前Python
    return sys.executable

python_executable = get_python_executable()

def run_hunyuan_cache(
    dataset_config_file: str,
    dataset_config_text: str,
    vae_path: str,
    text_encoder1_path: str,
    text_encoder2_path: str,
    enable_low_memory: bool,
    skip_existing: bool,
    use_clip: bool,
    clip_model_path: str
) -> Generator[str, None, None]:

    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)
    if not dataset_config:
        yield "[ERROR] 请提供数据集配置文件\n"
        return

    settings = {
        "hunyuan_cache": {
            "dataset_config_text": dataset_config_text,
            "vae_path": vae_path,
            "text_encoder1_path": text_encoder1_path,
            "text_encoder2_path": text_encoder2_path,
            "enable_low_memory": enable_low_memory,
            "skip_existing": skip_existing,
            "use_clip": use_clip,
            "clip_model_path": clip_model_path
        }
    }
    existing_settings = load_settings()
    existing_settings.update(settings)
    save_settings(existing_settings)

    cache_latents_cmd = [
        python_executable, "cache_latents.py",
        "--dataset_config", dataset_config,
        "--vae", vae_path,
        "--batch_size", "1"
    ]
    if enable_low_memory:
        cache_latents_cmd.extend(["--vae_spatial_tile_sample_min_size", "128", "--batch_size", "1"])
    if skip_existing:
        cache_latents_cmd.append("--skip_existing")
    if use_clip and clip_model_path.strip():
        cache_latents_cmd.extend(["--clip", clip_model_path.strip()])

    cache_text_encoder_cmd = [
        python_executable, "cache_text_encoder_outputs.py",
        "--dataset_config", dataset_config,
        "--text_encoder1", text_encoder1_path,
        "--text_encoder2", text_encoder2_path
    ]
    if enable_low_memory:
        cache_text_encoder_cmd.append("--fp8_llm")

    def run_and_stream_output(cmd):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        running_processes["cache"] = proc
        accumulated = ""
        for line in iter(proc.stdout.readline, ''):
            if line:
                accumulated += line
                yield accumulated
        proc.wait()
        running_processes["cache"] = None
        if proc.returncode != 0:
            accumulated += f"\n[ERROR] 进程退出代码 {proc.returncode}\n"
            yield accumulated

    accumulated_main = "\n[INFO] 开始 HunyuanVideo 潜在变量缓存 (cache_latents.py)...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_latents_cmd):
        yield content
    accumulated_main += "\n[INFO] HunyuanVideo 潜在变量缓存完成。\n"
    yield accumulated_main

    accumulated_main += "\n[INFO] 开始 HunyuanVideo 文本编码器缓存 (cache_text_encoder_outputs.py)...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_text_encoder_cmd):
        yield content
    accumulated_main += "\n[INFO] HunyuanVideo 文本编码器缓存完成。\n"
    yield accumulated_main

def run_hunyuan_training(
    dataset_config_file: str,
    dataset_config_text: str,
    dit_weights_path: str,
    max_train_epochs: int,
    learning_rate: str,
    network_dim: int,
    network_alpha: int,
    blocks_to_swap: int,
    output_dir: str,
    output_name: str,
    save_every_n_epochs: int,
    use_network_weights: bool,
    network_weights_path: str,
    gradient_checkpointing_cpu_offload: bool
) -> Generator[str, None, None]:

    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)
    if not dataset_config:
        yield "[ERROR] 请提供数据集配置文件\n"
        return

    settings = {
        "hunyuan_training": {
            "dataset_config_text": dataset_config_text,
            "dit_weights_path": dit_weights_path,
            "max_train_epochs": max_train_epochs,
            "learning_rate": learning_rate,
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "blocks_to_swap": blocks_to_swap,
            "output_dir": output_dir,
            "output_name": output_name,
            "save_every_n_epochs": save_every_n_epochs,
            "use_network_weights": use_network_weights,
            "network_weights_path": network_weights_path,
            "gradient_checkpointing_cpu_offload": gradient_checkpointing_cpu_offload
        }
    }
    existing_settings = load_settings()
    existing_settings.update(settings)
    save_settings(existing_settings)

    command = [
        python_executable, "-m", "accelerate.commands.launch",
        "--num_processes", "1",
        "--gpu_ids", "0",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16",
        "hv_train_network.py",
        "--dit", dit_weights_path,
        "--dataset_config", dataset_config,
        "--sdpa",
        "--mixed_precision", "bf16",
        "--fp8_base",
        "--optimizer_type", "adamw8bit",
        "--learning_rate", learning_rate,
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        "--network_module=src.musubi_tuner.networks.lora",
        f"--network_dim={network_dim}",
        f"--network_alpha={network_alpha}",
        "--timestep_sampling", "sigmoid",
        "--discrete_flow_shift", "1.0",
        "--max_train_epochs", str(max_train_epochs),
        "--seed", "42",
        "--output_dir", output_dir,
        "--output_name", output_name,
        "--save_every_n_epochs", str(save_every_n_epochs),
        "--save_model_as", "safetensors",
        f"--blocks_to_swap={blocks_to_swap}"
    ]

    if gradient_checkpointing_cpu_offload:
        command.append("--gradient_checkpointing_cpu_offload")

    if use_network_weights and network_weights_path.strip():
        command.extend(["--network_weights", network_weights_path.strip()])

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    running_processes["train"] = proc
    accumulated = "[INFO] 开始 HunyuanVideo LoRA 训练...\n\n"
    yield accumulated
    last_yield = time.time()
    for line in iter(proc.stdout.readline, ''):
        if line:
            accumulated += line
            if time.time() - last_yield >= 1.5:
                last_yield = time.time()
                yield accumulated
    proc.wait()
    running_processes["train"] = None
    if proc.returncode != 0:
        accumulated += f"\n[ERROR] 训练退出代码 {proc.returncode}\n"
    else:
        accumulated += "\n[INFO] 训练成功完成！\n"
    yield accumulated

def run_qwen_cache(
    dataset_config_file: str,
    dataset_config_text: str,
    vae_path: str,
    dit_path: str,
    text_encoder_path: str,
    enable_low_memory: bool,
    skip_existing: bool
) -> Generator[str, None, None]:

    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)
    if not dataset_config:
        yield "[ERROR] 请提供数据集配置文件\n"
        return

    settings = {
        "qwen_cache": {
            "dataset_config_text": dataset_config_text,
            "vae_path": vae_path,
            "dit_path": dit_path,
            "text_encoder_path": text_encoder_path,
            "enable_low_memory": enable_low_memory,
            "skip_existing": skip_existing
        }
    }
    existing_settings = load_settings()
    existing_settings.update(settings)
    save_settings(existing_settings)

    cache_latents_cmd = [
        python_executable, "qwen_image_cache_latents.py",
        "--dataset_config", dataset_config,
        "--vae", vae_path,
        "--dit", dit_path
    ]
    if skip_existing:
        cache_latents_cmd.append("--skip_existing")

    cache_text_encoder_cmd = [
        python_executable, "qwen_image_cache_text_encoder_outputs.py",
        "--dataset_config", dataset_config,
        "--text_encoder", text_encoder_path
    ]
    if enable_low_memory:
        cache_text_encoder_cmd.append("--fp8_vl")

    def run_and_stream_output(cmd):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=get_env_with_pythonpath())
        running_processes["cache"] = proc
        accumulated = ""
        for line in iter(proc.stdout.readline, ''):
            if line:
                accumulated += line
                yield accumulated
        proc.wait()
        running_processes["cache"] = None
        if proc.returncode != 0:
            accumulated += f"\n[ERROR] 进程退出代码 {proc.returncode}\n"
            yield accumulated

    accumulated_main = "\n[INFO] 开始 Qwen-Image 潜在变量缓存...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_latents_cmd):
        yield content
    accumulated_main += "\n[INFO] Qwen-Image 潜在变量缓存完成。\n"
    yield accumulated_main

    accumulated_main += "\n[INFO] 开始 Qwen-Image 文本编码器缓存...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_text_encoder_cmd):
        yield content
    accumulated_main += "\n[INFO] Qwen-Image 文本编码器缓存完成。\n"
    yield accumulated_main

def run_qwen_training(
    dataset_config_file: str,
    dataset_config_text: str,
    dit_weights_path: str,
    max_train_epochs: int,
    learning_rate: str,
    network_dim: int,
    network_alpha: int,
    output_dir: str,
    output_name: str,
    save_every_n_epochs: int,
    use_network_weights: bool,
    network_weights_path: str,
    enable_edit_mode: bool
) -> Generator[str, None, None]:

    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)
    if not dataset_config:
        yield "[ERROR] 请提供数据集配置文件\n"
        return

    settings = {
        "qwen_training": {
            "dataset_config_text": dataset_config_text,
            "dit_weights_path": dit_weights_path,
            "max_train_epochs": max_train_epochs,
            "learning_rate": learning_rate,
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "output_dir": output_dir,
            "output_name": output_name,
            "save_every_n_epochs": save_every_n_epochs,
            "use_network_weights": use_network_weights,
            "network_weights_path": network_weights_path,
            "enable_edit_mode": enable_edit_mode
        }
    }
    existing_settings = load_settings()
    existing_settings.update(settings)
    save_settings(existing_settings)

    command = [
        python_executable, "-m", "accelerate.commands.launch",
        "--num_processes", "1",
        "--gpu_ids", "0",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16",
        "qwen_image_train_network.py",
        "--dit", dit_weights_path,
        "--dataset_config", dataset_config,
        "--sdpa",
        "--mixed_precision", "bf16",
        "--optimizer_type", "adamw8bit",
        "--learning_rate", learning_rate,
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        "--network_module=src.musubi_tuner.networks.lora_qwen_image",
        f"--network_dim={network_dim}",
        f"--network_alpha={network_alpha}",
        "--max_train_epochs", str(max_train_epochs),
        "--seed", "42",
        "--output_dir", output_dir,
        "--output_name", output_name,
        "--save_every_n_epochs", str(save_every_n_epochs),
        "--save_model_as", "safetensors"
    ]

    if enable_edit_mode:
        command.append("--edit")

    if use_network_weights and network_weights_path.strip():
        command.extend(["--network_weights", network_weights_path.strip()])

    # 清空之前的训练数据
    training_data['steps'].clear()
    training_data['losses'].clear()
    training_data['timestamps'].clear()
    training_data['learning_rates'].clear()
    training_data['epochs'].clear()

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=get_env_with_pythonpath())
    running_processes["train"] = proc
    accumulated = "[INFO] 开始 Qwen-Image LoRA 训练...\n\n"
    yield accumulated
    last_yield = time.time()
    for line in iter(proc.stdout.readline, ''):
        if line:
            accumulated += line
            parsed_data = parse_training_log(line)
            if parsed_data:
                if len(training_data['learning_rates']) > 0:
                    training_data['learning_rates'][-1] = float(learning_rate) if learning_rate else 1e-4
            if time.time() - last_yield >= 1.5:
                last_yield = time.time()
                yield accumulated
    proc.wait()
    running_processes["train"] = None
    if proc.returncode != 0:
        accumulated += f"\n[ERROR] 训练退出代码 {proc.returncode}\n"
    else:
        accumulated += "\n[INFO] 训练成功完成！\n"
    yield accumulated

def run_wan_cache(
    dataset_config_file: str,
    dataset_config_text: str,
    enable_low_memory: bool,
    skip_existing: bool,
    vae_path: str,
    t5_path: str,
    enable_i2v: bool,
    clip_model_path: str
) -> Generator[str, None, None]:

    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)
    if not dataset_config:
        yield "[ERROR] 请提供数据集配置文件\n"
        return

    settings_to_save = {
        "wan_cache": {
            "dataset_config_text": dataset_config_text,
            "enable_low_memory": enable_low_memory,
            "skip_existing": skip_existing,
            "vae_path": vae_path,
            "t5_path": t5_path,
            "enable_i2v": enable_i2v,
            "clip_path": clip_model_path
        }
    }
    existing_settings = load_settings()
    existing_settings.update(settings_to_save)
    save_settings(existing_settings)

    cache_latents_cmd = [
        python_executable, "wan_cache_latents.py",
        "--dataset_config", dataset_config,
        "--vae", vae_path
    ]
    if enable_low_memory:
        cache_latents_cmd.append("--vae_cache_cpu")
    if skip_existing:
        cache_latents_cmd.append("--skip_existing")
    if enable_i2v and clip_model_path and clip_model_path.strip():
        cache_latents_cmd.extend(["--clip", clip_model_path.strip()])

    cache_text_encoder_cmd = [
        python_executable, "wan_cache_text_encoder_outputs.py",
        "--dataset_config", dataset_config,
        "--t5", t5_path,
        "--batch_size", "16"
    ]
    if enable_low_memory:
        cache_text_encoder_cmd.append("--fp8_t5")
    if skip_existing:
        cache_text_encoder_cmd.append("--skip_existing")

    accumulated_main = "[INFO] 开始 Wan 潜在变量缓存 (wan_cache_latents.py)...\n\n"
    yield accumulated_main

    proc = subprocess.Popen(cache_latents_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=get_env_with_pythonpath())
    running_processes["cache"] = proc
    for line in iter(proc.stdout.readline, ''):
        if line:
            accumulated_main += line
            yield accumulated_main
    proc.wait()
    running_processes["cache"] = None
    if proc.returncode != 0:
        accumulated_main += f"\n[ERROR] 潜在变量缓存退出代码 {proc.returncode}\n"
        yield accumulated_main
        return

    accumulated_main += "\n[INFO] Wan 潜在变量缓存完成。\n\n"
    yield accumulated_main

    accumulated_main += "[INFO] 开始 Wan 文本编码器缓存 (wan_cache_text_encoder_outputs.py)...\n\n"
    yield accumulated_main

    proc = subprocess.Popen(cache_text_encoder_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=get_env_with_pythonpath())
    running_processes["cache"] = proc
    for line in iter(proc.stdout.readline, ''):
        if line:
            accumulated_main += line
            yield accumulated_main
    proc.wait()
    running_processes["cache"] = None
    if proc.returncode != 0:
        accumulated_main += f"\n[ERROR] 文本编码器缓存退出代码 {proc.returncode}\n"
        yield accumulated_main
        return

    accumulated_main += "\n[INFO] Wan 文本编码器缓存完成。\n"
    yield accumulated_main

def run_wan_training(
    task: str,
    dataset_config_file: str,
    dataset_config_text: str,
    dit_weights_path: str,
    is_wan22: bool,
    dit_low_noise_path: str,
    max_train_epochs: int,
    learning_rate: str,
    network_dim: int,
    network_alpha: int,
    blocks_to_swap: int,
    fp8: bool,
    output_dir: str,
    output_name: str,
    save_every_n_epochs: int,
    use_network_weights: bool,
    network_weights_path: str,
    attention_mode: str,
    mixed_precision: str,
    optimizer_type: str,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    lr_scheduler: str,
    lr_warmup_steps: int,
    timestep_sampling: str,
    discrete_flow_shift: float,
    weighting_scheme: str,
    enable_gradient_checkpointing: bool,
    seed: int,
    sample_every_n_epochs: int,
    sample_prompts: str,
    sample_steps: int,
    sample_solver: str,
    logging_dir: str,
    wandb_run_name: str
) -> Generator[str, None, None]:

    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)
    if not dataset_config:
        yield "[ERROR] 请提供数据集配置文件\n"
        return

    settings = {
        "wan_training": {
            "task": task,
            "dataset_config_text": dataset_config_text,
            "dit_weights_path": dit_weights_path,
            "is_wan22": is_wan22,
            "dit_low_noise_path": dit_low_noise_path,
            "max_train_epochs": max_train_epochs,
            "learning_rate": learning_rate,
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "blocks_to_swap": blocks_to_swap,
            "fp8": fp8,
            "output_dir": output_dir,
            "output_name": output_name,
            "save_every_n_epochs": save_every_n_epochs,
            "use_network_weights": use_network_weights,
            "network_weights_path": network_weights_path,
            "attention_mode": attention_mode,
            "mixed_precision": mixed_precision,
            "optimizer_type": optimizer_type,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_grad_norm": max_grad_norm,
            "lr_scheduler": lr_scheduler.split(" - ")[0] if " - " in lr_scheduler else lr_scheduler,
            "lr_warmup_steps": lr_warmup_steps,
            "timestep_sampling": timestep_sampling.split(" - ")[0] if " - " in timestep_sampling else timestep_sampling,
            "discrete_flow_shift": discrete_flow_shift,
            "weighting_scheme": weighting_scheme.split(" - ")[0] if " - " in weighting_scheme else weighting_scheme,
            "enable_gradient_checkpointing": enable_gradient_checkpointing,
            "seed": seed,
            "sample_every_n_epochs": sample_every_n_epochs,
            "sample_prompts": sample_prompts,
            "sample_steps": sample_steps,
            "sample_solver": sample_solver,
            "logging_dir": logging_dir,
            "wandb_run_name": wandb_run_name
        }
    }
    existing_settings = load_settings()
    existing_settings.update(settings)
    save_settings(existing_settings)

    command = [
        python_executable, "-m", "accelerate.commands.launch",
        "--num_processes", "1",
        "--gpu_ids", "0",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", mixed_precision,
        "wan_train_network.py",
        "--task", task,
        "--dit", dit_weights_path,
        "--dataset_config", dataset_config,
        "--mixed_precision", mixed_precision,
        "--optimizer_type", optimizer_type,
        "--learning_rate", learning_rate,
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        "--network_module=src.musubi_tuner.networks.lora_wan",
        f"--network_dim={network_dim}",
        f"--network_alpha={network_alpha}",
        "--max_train_epochs", str(max_train_epochs),
        "--seed", str(seed),
        "--output_dir", output_dir,
        "--output_name", output_name,
        "--save_every_n_epochs", str(save_every_n_epochs),
        "--blocks_to_swap", str(blocks_to_swap),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--max_grad_norm", str(max_grad_norm),
        "--lr_scheduler", lr_scheduler.split(" - ")[0] if " - " in lr_scheduler else lr_scheduler,
        "--lr_warmup_steps", str(lr_warmup_steps),
        "--timestep_sampling", timestep_sampling.split(" - ")[0] if " - " in timestep_sampling else timestep_sampling,
        "--discrete_flow_shift", str(discrete_flow_shift),
        "--weighting_scheme", weighting_scheme.split(" - ")[0] if " - " in weighting_scheme else weighting_scheme
    ]

    if attention_mode == "sdpa":
        command.append("--sdpa")
    elif attention_mode == "flash_attn":
        command.append("--flash_attn")
    elif attention_mode == "sage_attn":
        command.append("--sage_attn")
    elif attention_mode == "xformers":
        command.append("--xformers")

    if enable_gradient_checkpointing:
        command.append("--gradient_checkpointing")

    if fp8:
        command.append("--fp8_base")

    if is_wan22 and dit_low_noise_path.strip():
        command.extend(["--dit_high_noise", dit_low_noise_path.strip()])

    if use_network_weights and network_weights_path.strip():
        command.extend(["--network_weights", network_weights_path.strip()])

    command.extend(["--sample_steps", str(sample_steps)])
    command.extend(["--sample_solver", sample_solver])

    if sample_every_n_epochs > 0 and sample_prompts.strip():
        temp_prompts_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
        temp_prompts_file.write(sample_prompts.strip())
        temp_prompts_file.close()

        command.extend(["--sample_every_n_epochs", str(sample_every_n_epochs)])
        command.extend(["--sample_prompts", temp_prompts_file.name])

    if logging_dir.strip():
        command.extend(["--logging_dir", logging_dir.strip()])
        command.append("--log_with=tensorboard")

    if wandb_run_name.strip():
        command.extend(["--wandb_run_name", wandb_run_name.strip()])
        command.append("--log_with=wandb")

    # 清空之前的训练数据
    training_data['steps'].clear()
    training_data['losses'].clear()
    training_data['timestamps'].clear()
    training_data['learning_rates'].clear()
    training_data['epochs'].clear()

    accumulated = f"[INFO] 开始 Wan LoRA 训练 (任务: {task})...\n"
    accumulated += f"[DEBUG] 采样步数: {sample_steps}, 采样器: {sample_solver}\n"
    accumulated += f"[DEBUG] 完整命令: {' '.join(command)}\n\n"
    yield accumulated

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=get_env_with_pythonpath())
    running_processes["train"] = proc
    last_yield = time.time()
    for line in iter(proc.stdout.readline, ''):
        if line:
            accumulated += line
            parsed_data = parse_training_log(line)
            if parsed_data:
                if len(training_data['learning_rates']) > 0:
                    training_data['learning_rates'][-1] = float(learning_rate) if learning_rate else 1e-5
            if time.time() - last_yield >= 1.5:
                last_yield = time.time()
                yield accumulated
    proc.wait()
    running_processes["train"] = None
    if proc.returncode != 0:
        accumulated += f"\n[ERROR] 训练退出代码 {proc.returncode}\n"
    else:
        accumulated += "\n[INFO] 训练成功完成！\n"
    yield accumulated

settings = load_settings()



custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 50%, #e9ecef 100%) !important;
}
.tabs {
    background: rgba(0, 0, 0, 0.05) !important;
    border-radius: 12px !important;
    padding: 10px !important;
}
.tab-nav button {
    background: rgba(0, 0, 0, 0.08) !important;
    color: #2c3e50 !important;
    border: 1px solid rgba(0, 0, 0, 0.15) !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    margin: 0 4px !important;
    border-radius: 8px 8px 0 0 !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
    border-bottom: 3px solid #2c3e50 !important;
    color: white !important;
}
label {
    color: #2c3e50 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
.markdown {
    color: #2c3e50 !important;
}
h1 {
    color: #2980b9 !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1) !important;
}
h2, h3 {
    color: #34495e !important;
}
.primary {
    background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
}
.stop {
    background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
}
input:not([type="checkbox"]):not([type="radio"]), textarea, select {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(0, 0, 0, 0.2) !important;
    color: #2c3e50 !important;
    border-radius: 6px !important;
}
input[type="checkbox"], input[type="radio"] {
    accent-color: #3498db !important;
    cursor: pointer !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Musubi Tuner v0.2.13") as demo:
    gr.Markdown("""
    # Musubi Tuner v0.2.13 专业训练界面
    ### 界面作者: suzuki & eddy | 基于 Kohya's Musubi Tuner
    **支持模型:** HunyuanVideo | Wan2.1/2.2 | Qwen-Image | FramePack | FLUX Kontext
    ---
    """)

    with gr.Tab("快速开始"):
        gr.Markdown("""
        ## 欢迎使用 Musubi Tuner v0.2.13

        ### 支持的模型:
        - **HunyuanVideo**: 文本转视频 & 图像转视频
        - **Wan2.1/2.2**: 1.3B & 14B 模型
        - **Qwen-Image**: 文本转图像 & 图像编辑
        - **FramePack**: 渐进式图像转视频
        - **FLUX Kontext**: 高级图像生成

        ### 训练步骤:
        1. **准备数据集** - 创建 TOML 配置文件
        2. **预缓存** - 缓存潜在变量和文本编码器输出
        3. **训练** - 开始 LoRA 训练
        4. **生成** - 使用训练好的 LoRA 进行推理

        ### 模型下载链接:
        - **HunyuanVideo**: [hunyuanvideo-community/HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo)
        - **Wan2.1**: [eddy1111111/WAN_train_models](https://huggingface.co/eddy1111111/WAN_train_models)
        - **Wan2.2**: [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)
        - **Qwen-Image**: [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI)
        - **FramePack**: [lllyasviel/FramePackI2V_HY](https://huggingface.co/lllyasviel/FramePackI2V_HY)

        ### 性能优化提示:
        - **FP8 量化**: 节省显存
        - **CPU 卸载**: 节省 20-30% 显存
        - **网络维度**: 从 16-32 开始以加快训练
        - **块交换**: 使用 16-20 块来减少显存使用
        - **低内存模式**: 为 16GB 以下显存的 GPU 启用

        ### 模型路径示例:
        ```
        VAE: ./models/hunyuan/vae/diffusion_pytorch_model.safetensors
        Text Encoder 1: ./models/hunyuan/text_encoder/model-00001-of-00004.safetensors
        Text Encoder 2: ./models/hunyuan/text_encoder_2/model.safetensors
        DiT: ./models/hunyuan/hunyuan_video_fp8_scaled.safetensors
        ```
        """)
        gr.Markdown("""
        ### 模型放置与路径书写规范（跨平台）
        - 建议使用相对路径，如: ./models/...
        - 使用正斜杠 /，兼容 Windows/Linux/macOS
        - 路径尽量避免中文与空格（包含空格也可使用）
        - 推荐目录结构：
        ```
        ./models/
          qwen/
            qwen_image_bf16.safetensors
            qwen_image_vae.safetensors
            Qwen2.5-VL-7B-Instruct/
          wan/
            wan_2.1_vae.safetensors
            models_t5_umt5-xxl-enc-bf16.pth
            models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
            wan_2.1_t2v_14b_bf16.safetensors
            wan_2.2_low_noise_bf16.safetensors
        ./datasets/
          my_dataset.toml
        ./output/
        ```
        """)


    with gr.Tab("Qwen-Image"):
        gr.Markdown("## Qwen-Image LoRA 训练")
        with gr.Tabs():
            with gr.Tab("预缓存"):
                gr.Markdown("""
                ### 步骤 1: 缓存潜在变量和文本编码器输出
                **所需模型:**
                - DiT: `qwen_image_bf16.safetensors` (必须使用 BF16 版本，不能使用 FP8)
                - VAE: `qwen_image_vae.safetensors`
                - Text Encoder: Qwen2.5-VL 模型
                - 下载: [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI)
                """)
                gr.Markdown("""
                #### 路径与目录示例（跨平台）
                - VAE: ./models/qwen/qwen_image_vae.safetensors
                - DiT (BF16): ./models/qwen/qwen_image_bf16.safetensors
                - Text Encoder 目录: ./models/qwen/Qwen2.5-VL-7B-Instruct
                - 数据集 TOML: ./datasets/my_dataset.toml
                - 建议使用正斜杠 /，避免转义
                """)
                with gr.Row():
                    qw_cache_dataset_file = gr.File(label="上传数据集配置 (TOML)", file_count="single", file_types=[".toml"], type="filepath")
                    qw_cache_dataset_text = gr.Textbox(label="或输入 TOML 路径", placeholder="./datasets/my_dataset.toml", value=settings.get("qwen_cache", {}).get("dataset_config_text", ""), interactive=True)
                qw_cache_low_memory = gr.Checkbox(label="启用低内存模式 (FP8 Text Encoder)", value=settings.get("qwen_cache", {}).get("enable_low_memory", False), interactive=True)
                qw_cache_skip_existing = gr.Checkbox(label="跳过已存在的缓存文件", value=settings.get("qwen_cache", {}).get("skip_existing", True), interactive=True)
                qw_cache_vae = gr.Textbox(label="VAE 模型路径", placeholder="./models/qwen/qwen_image_vae.safetensors", value=settings.get("qwen_cache", {}).get("vae_path", ""), interactive=True)
                qw_cache_dit = gr.Textbox(label="DiT 模型路径 (必须 BF16)", placeholder="./models/qwen/qwen_image_bf16.safetensors", value=settings.get("qwen_cache", {}).get("dit_path", ""), interactive=True)
                qw_cache_te = gr.Textbox(label="Text Encoder 路径 (Qwen2.5-VL)", placeholder="./models/qwen/Qwen2.5-VL-7B-Instruct", value=settings.get("qwen_cache", {}).get("text_encoder_path", ""), interactive=True)
                with gr.Row():
                    qw_cache_run_btn = gr.Button("开始预缓存", variant="primary", size="lg")
                    qw_cache_stop_btn = gr.Button("停止", variant="stop", size="lg")
                qw_cache_output = gr.Textbox(label="缓存输出日志", lines=20, interactive=False, show_copy_button=True)
                qw_cache_run_btn.click(fn=run_qwen_cache, inputs=[qw_cache_dataset_file, qw_cache_dataset_text, qw_cache_vae, qw_cache_dit, qw_cache_te, qw_cache_low_memory, qw_cache_skip_existing], outputs=qw_cache_output)
                qw_cache_stop_btn.click(fn=stop_caching, outputs=qw_cache_output)

            with gr.Tab("训练"):
                gr.Markdown("""
                ### 步骤 2: 训练 LoRA 模型
                **所需模型:**
                - DiT: `qwen_image_bf16.safetensors` (必须使用 BF16 版本)
                - 下载: [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI)

                **模式说明:**
                - **文本转图像 (T2I)**: 标准图像生成训练
                - **图像编辑模式**: 启用后支持图像编辑功能 (Edit-2509)
                """)
                gr.Markdown("""
                #### 路径与目录示例（跨平台）
                - DiT (BF16): ./models/qwen/qwen_image_bf16.safetensors
                - 输出目录: ./output
                - 继续训练的 LoRA: ./output/qwen_lora.safetensors
                - 数据集 TOML: ./datasets/my_dataset.toml
                - 建议使用相对路径和正斜杠 /，兼容 Windows/Linux/macOS
                """)
                with gr.Row():
                    qw_train_dataset_file = gr.File(label="上传数据集配置 (TOML)", file_count="single", file_types=[".toml"], type="filepath")
                    qw_train_dataset_text = gr.Textbox(label="或输入 TOML 路径", placeholder="./datasets/my_dataset.toml", value=settings.get("qwen_training", {}).get("dataset_config_text", ""), interactive=True)
                qw_train_dit = gr.Textbox(label="DiT 模型路径 (必须 BF16)", placeholder="./models/qwen/qwen_image_bf16.safetensors", value=settings.get("qwen_training", {}).get("dit_weights_path", ""), interactive=True)
                with gr.Row():
                    qw_train_epochs = gr.Number(label="训练轮数", value=settings.get("qwen_training", {}).get("max_train_epochs", 16), precision=0, minimum=1, interactive=True)
                    qw_train_lr = gr.Textbox(label="学习率", value=settings.get("qwen_training", {}).get("learning_rate", "1e-4"), placeholder="1e-4", interactive=True)
                with gr.Row():
                    qw_train_network_dim = gr.Number(label="网络维度 (LoRA rank)", value=settings.get("qwen_training", {}).get("network_dim", 32), precision=0, minimum=1, interactive=True)
                    qw_train_network_alpha = gr.Number(label="网络 Alpha", value=settings.get("qwen_training", {}).get("network_alpha", 16), precision=0, minimum=1, interactive=True)
                qw_train_edit_mode = gr.Checkbox(label="启用图像编辑模式 (Edit-2509)", value=settings.get("qwen_training", {}).get("enable_edit_mode", False), interactive=True)
                with gr.Row():
                    qw_train_output_dir = gr.Textbox(label="输出目录", value=settings.get("qwen_training", {}).get("output_dir", "./output"), placeholder="./output", interactive=True)
                    qw_train_output_name = gr.Textbox(label="输出名称", value=settings.get("qwen_training", {}).get("output_name", "qwen_lora"), placeholder="qwen_lora", interactive=True)
                qw_train_save_every = gr.Number(label="每 N 轮保存一次", value=settings.get("qwen_training", {}).get("save_every_n_epochs", 2), precision=0, minimum=1, interactive=True)
                with gr.Row():
                    qw_train_use_network_weights = gr.Checkbox(label="继续训练 (加载已有 LoRA)", value=settings.get("qwen_training", {}).get("use_network_weights", False), interactive=True)
                    qw_train_network_weights_path = gr.Textbox(label="已有 LoRA 路径", placeholder="./output/qwen_lora.safetensors", visible=settings.get("qwen_training", {}).get("use_network_weights", False), value=settings.get("qwen_training", {}).get("network_weights_path", ""), interactive=True)
                qw_train_use_network_weights.change(lambda x: gr.update(visible=x), inputs=qw_train_use_network_weights, outputs=qw_train_network_weights_path)
                with gr.Row():
                    qw_train_run_btn = gr.Button("开始训练", variant="primary", size="lg")
                    qw_train_stop_btn = gr.Button("停止", variant="stop", size="lg")
                qw_train_output = gr.Textbox(label="训练输出日志", lines=20, interactive=False, show_copy_button=True)

                # 添加实时训练曲线图
                gr.Markdown("### 📊 实时训练监控")
                with gr.Row():
                    with gr.Column():
                        qw_loss_plot = gr.Plot(label="Loss 曲线", value=create_loss_plot())
                    with gr.Column():
                        qw_progress_plot = gr.Plot(label="训练进度 & 学习率", value=create_progress_plot())

                # 创建定时更新函数
                def update_qw_plots():
                    return create_loss_plot(), create_progress_plot()

                # 设置定时器更新图表（每2秒更新一次，更实时）
                qw_plot_timer = gr.Timer(2)
                qw_plot_timer.tick(fn=update_qw_plots, outputs=[qw_loss_plot, qw_progress_plot])

                qw_train_run_btn.click(fn=run_qwen_training, inputs=[qw_train_dataset_file, qw_train_dataset_text, qw_train_dit, qw_train_epochs, qw_train_lr, qw_train_network_dim, qw_train_network_alpha, qw_train_output_dir, qw_train_output_name, qw_train_save_every, qw_train_use_network_weights, qw_train_network_weights_path, qw_train_edit_mode], outputs=qw_train_output)
                qw_train_stop_btn.click(fn=stop_training, outputs=qw_train_output)

    with gr.Tab("Wan2.1/2.2"):
        gr.Markdown("## Wan2.1/2.2 LoRA 训练")
        gr.Markdown("""
        **注意**: Wan2.1 和 Wan2.2 使用相同的训练流程
        - Wan2.1: 支持 T2V 和 I2V
        - Wan2.2: 仅支持 14B 模型，使用双 DiT 架构 (高噪声 + 低噪声)
        """)
        with gr.Tabs():
            with gr.Tab("预缓存"):
                gr.Markdown("""
                ### 步骤 1: 缓存潜在变量和文本编码器输出
                **所需模型:**
                - VAE: `wan_2.1_vae.safetensors` 或 `Wan2.1_VAE.pth`
                - T5: `models_t5_umt5-xxl-enc-bf16.pth`
                - CLIP: `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` (仅 Wan2.1 需要)
                - 下载: [eddy1111111/WAN_train_models](https://huggingface.co/eddy1111111/WAN_train_models)
                """)
                gr.Markdown("""
                ####






                ####
                ####
                ####
                ####
""")
                gr.Markdown("""
                #### 路径与目录示例（跨平台）
                - VAE: ./models/wan/wan_2.1_vae.safetensors
                - T5: ./models/wan/models_t5_umt5-xxl-enc-bf16.pth
                - 数据集 TOML: ./datasets/my_dataset.toml
                - 建议使用相对路径和正斜杠 /，兼容 Windows/Linux/macOS

                **注意**: 大部分用户使用 T2V 模式，不需要启用 I2V 选项
                """)
                with gr.Row():
                    wan_cache_dataset_file = gr.File(label="上传数据集配置 (TOML)", file_count="single", file_types=[".toml"], type="filepath")
                    wan_cache_dataset_text = gr.Textbox(label="或输入 TOML 路径", placeholder="./datasets/my_dataset.toml", value=settings.get("wan_cache", {}).get("dataset_config_text", ""), interactive=True)
                wan_cache_low_memory = gr.Checkbox(label="启用低内存模式 (FP8 T5)", value=settings.get("wan_cache", {}).get("enable_low_memory", False), interactive=True)
                wan_cache_skip_existing = gr.Checkbox(label="跳过已存在的缓存文件", value=settings.get("wan_cache", {}).get("skip_existing", True), interactive=True)
                wan_cache_vae = gr.Textbox(label="VAE 模型路径", placeholder="./models/wan/wan_2.1_vae.safetensors", value=settings.get("wan_cache", {}).get("vae_path", ""), interactive=True)
                wan_cache_t5 = gr.Textbox(label="T5 模型路径", placeholder="./models/wan/models_t5_umt5-xxl-enc-bf16.pth", value=settings.get("wan_cache", {}).get("t5_path", ""), interactive=True)
                wan_cache_enable_i2v = gr.Checkbox(label="启用 I2V 模式 (需要 CLIP 模型，仅 Wan2.1)", value=settings.get("wan_cache", {}).get("enable_i2v", False), interactive=True)
                wan_cache_clip = gr.Textbox(label="CLIP 模型路径", placeholder="./models/wan/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", value=settings.get("wan_cache", {}).get("clip_path", ""), visible=settings.get("wan_cache", {}).get("enable_i2v", False), interactive=True)
                wan_cache_enable_i2v.change(lambda x: gr.update(visible=x), inputs=wan_cache_enable_i2v, outputs=wan_cache_clip)
                with gr.Row():
                    wan_cache_run_btn = gr.Button("开始预缓存", variant="primary", size="lg")
                    wan_cache_stop_btn = gr.Button("停止", variant="stop", size="lg")
                wan_cache_output = gr.Textbox(label="缓存输出日志", lines=20, interactive=False, show_copy_button=True)
                wan_cache_run_btn.click(
                    fn=run_wan_cache,
                    inputs=[
                        wan_cache_dataset_file,
                        wan_cache_dataset_text,
                        wan_cache_low_memory,
                        wan_cache_skip_existing,
                        wan_cache_vae,
                        wan_cache_t5,
                        wan_cache_enable_i2v,
                        wan_cache_clip,
                    ],
                    outputs=wan_cache_output,
                )
                wan_cache_stop_btn.click(fn=stop_caching, outputs=wan_cache_output)

            with gr.Tab("训练"):
                gr.Markdown("""
                ### 步骤 2: 训练 LoRA 模型
                **所需模型:**
                - Wan2.1 DiT: 从 [eddy1111111/WAN_train_models](https://huggingface.co/eddy1111111/WAN_train_models) 下载
                - Wan2.2 DiT: 从 [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged) 下载
                - 支持 fp16, bf16, fp8_e4m3fn 模型 (不支持 fp8_scaled)

                **任务模式说明:**
                - **t2v-1.3B / t2v-14B**: Wan2.1 文本转视频 (1.3B / 14B 参数)
                - **i2v-14B**: Wan2.1 图像转视频 (需要 CLIP 模型)
                - **t2i-14B**: Wan2.1 文本转图像
                - **t2v-1.3B-FC / t2v-14B-FC / i2v-14B-FC**: Wan2.1 Fun Control 模型
                - **t2v-A14B / i2v-A14B**: Wan2.2 双 DiT 模型 (需要高噪声和低噪声两个 DiT)
                """)
                gr.Markdown("""
                #### 路径与目录示例（跨平台）
                - Wan2.1 DiT: ./models/wan/wan_2.1_t2v_14b_bf16.safetensors
                - Wan2.2 低噪声 DiT: ./models/wan/wan_2.2_low_noise_bf16.safetensors
                - 输出目录: ./output
                - 继续训练的 LoRA: ./output/wan_lora.safetensors
                - 建议使用相对路径和正斜杠 /，兼容 Windows/Linux/macOS
                """)
                wan_train_task = gr.Dropdown(
                    label="任务模式 (Task)",
                    choices=[
                        "t2v-1.3B", "t2v-14B", "i2v-14B", "t2i-14B", "flf2v-14B",
                        "t2v-1.3B-FC", "t2v-14B-FC", "i2v-14B-FC",
                        "t2v-A14B", "i2v-A14B"
                    ],
                    value=settings.get("wan_training", {}).get("task", "t2v-14B"),
                    info="Wan2.1: t2v/i2v/t2i/flf2v-1.3B/14B, FC=Fun Control | Wan2.2: t2v/i2v-A14B",
                    interactive=True
                )
                with gr.Row():
                    wan_train_dataset_file = gr.File(label="上传数据集配置 (TOML)", file_count="single", file_types=[".toml"], type="filepath")
                    wan_train_dataset_text = gr.Textbox(label="或输入 TOML 路径", placeholder="./datasets/my_dataset.toml", value=settings.get("wan_training", {}).get("dataset_config_text", ""), interactive=True)
                wan_train_dit = gr.Textbox(label="DiT 模型路径", placeholder="./models/wan/wan_2.1_t2v_14b_bf16.safetensors", value=settings.get("wan_training", {}).get("dit_weights_path", ""), interactive=True)
                wan_train_is_wan22 = gr.Checkbox(label="使用 Wan2.2 (双 DiT 架构)", value=settings.get("wan_training", {}).get("is_wan22", False), interactive=True)
                wan_train_dit_low_noise = gr.Textbox(label="DiT 低噪声模型路径 (仅 Wan2.2)", placeholder="./models/wan/wan_2.2_low_noise_bf16.safetensors", visible=settings.get("wan_training", {}).get("is_wan22", False), value=settings.get("wan_training", {}).get("dit_low_noise_path", ""), interactive=True)
                wan_train_is_wan22.change(lambda x: gr.update(visible=x), inputs=wan_train_is_wan22, outputs=wan_train_dit_low_noise)
                with gr.Row():
                    wan_train_epochs = gr.Number(label="训练轮数", value=settings.get("wan_training", {}).get("max_train_epochs", 16), precision=0, minimum=1, interactive=True)
                    wan_train_lr = gr.Textbox(label="学习率", value=settings.get("wan_training", {}).get("learning_rate", "1e-5"), placeholder="1e-5", interactive=True)
                with gr.Row():
                    wan_train_network_dim = gr.Number(label="网络维度 (LoRA rank)", value=settings.get("wan_training", {}).get("network_dim", 32), precision=0, minimum=1, interactive=True)
                    wan_train_network_alpha = gr.Number(label="网络 Alpha", value=settings.get("wan_training", {}).get("network_alpha", 16), precision=0, minimum=1, interactive=True)
                with gr.Row():
                    wan_train_blocks_to_swap = gr.Number(label="块交换数量 (0-36, 推荐: 16)", value=settings.get("wan_training", {}).get("blocks_to_swap", 16), precision=0, minimum=0, maximum=36, interactive=True)
                    wan_train_fp8 = gr.Checkbox(label="启用 FP8 量化", value=settings.get("wan_training", {}).get("fp8", True), interactive=True)
                with gr.Row():
                    wan_train_output_dir = gr.Textbox(label="输出目录", value=settings.get("wan_training", {}).get("output_dir", "./output"), placeholder="./output", interactive=True)
                    wan_train_output_name = gr.Textbox(label="输出名称", value=settings.get("wan_training", {}).get("output_name", "wan_lora"), placeholder="wan_lora", interactive=True)
                wan_train_save_every = gr.Number(label="每 N 轮保存一次", value=settings.get("wan_training", {}).get("save_every_n_epochs", 2), precision=0, minimum=1, interactive=True)
                with gr.Row():
                    wan_train_use_network_weights = gr.Checkbox(label="继续训练 (加载已有 LoRA)", value=settings.get("wan_training", {}).get("use_network_weights", False), interactive=True)
                    wan_train_network_weights_path = gr.Textbox(label="已有 LoRA 路径", placeholder="./output/wan_lora.safetensors", visible=settings.get("wan_training", {}).get("use_network_weights", False), value=settings.get("wan_training", {}).get("network_weights_path", ""), interactive=True)
                wan_train_use_network_weights.change(lambda x: gr.update(visible=x), inputs=wan_train_use_network_weights, outputs=wan_train_network_weights_path)
                with gr.Row():
                    wan_train_run_btn = gr.Button("开始训练", variant="primary", size="lg")
                    wan_train_stop_btn = gr.Button("停止", variant="stop", size="lg")
                wan_train_output = gr.Textbox(label="训练输出日志", lines=20, interactive=False, show_copy_button=True)

            with gr.Tab("高级训练选项"):
                gr.Markdown("""
                ### 高级训练参数配置
                **注意**: 这些参数会影响训练质量和速度，建议有经验的用户调整
                """)

                gr.Markdown("#### 注意力机制")
                wan_adv_attention = gr.Dropdown(
                    label="注意力计算方式",
                    choices=["sdpa", "flash_attn", "sage_attn", "xformers"],
                    value=settings.get("wan_training", {}).get("attention_mode", "sage_attn"),
                    info="sdpa=PyTorch原生 | flash_attn=FlashAttention | sage_attn=SageAttention推荐 | xformers=xFormers",
                    interactive=True
                )

                gr.Markdown("#### 精度与优化器")
                with gr.Row():
                    wan_adv_mixed_precision = gr.Dropdown(
                        label="混合精度",
                        choices=["no", "fp16", "bf16"],
                        value=settings.get("wan_training", {}).get("mixed_precision", "bf16"),
                        info="bf16推荐用于RTX 30/40/50系列",
                        interactive=True
                    )
                    wan_adv_optimizer = gr.Dropdown(
                        label="优化器类型",
                        choices=["AdamW", "AdamW8bit", "AdaFactor", "Prodigy"],
                        value=settings.get("wan_training", {}).get("optimizer_type", "adamw8bit"),
                        info="AdamW8bit可节省显存",
                        interactive=True
                    )

                gr.Markdown("#### 梯度与学习率")
                with gr.Row():
                    wan_adv_grad_accum = gr.Number(
                        label="梯度累积步数",
                        value=settings.get("wan_training", {}).get("gradient_accumulation_steps", 1),
                        precision=0,
                        minimum=1,
                        info="增加此值可模拟更大的batch size",
                        interactive=True
                    )
                    wan_adv_max_grad_norm = gr.Number(
                        label="最大梯度范数",
                        value=settings.get("wan_training", {}).get("max_grad_norm", 1.0),
                        minimum=0,
                        info="梯度裁剪，0表示不裁剪",
                        interactive=True
                    )

                with gr.Row():
                    wan_adv_lr_scheduler = gr.Dropdown(
                        label="学习率调度器",
                        choices=[
                            "constant - 恒定学习率，适合微调和小数据集",
                            "constant_with_warmup - 恒定+预热，适合大模型训练",
                            "cosine - 余弦退火，收敛平滑，适合长时间训练",
                            "cosine_with_restarts - 余弦+重启，周期性重启，适合跳出局部最优",
                            "linear - 线性衰减，学习率线性下降到0",
                            "polynomial - 多项式衰减，介于线性和余弦之间"
                        ],
                        value=settings.get("wan_training", {}).get("lr_scheduler", "constant - 恒定学习率，适合微调和小数据集"),
                        info="学习率变化策略",
                        interactive=True
                    )
                    wan_adv_lr_warmup = gr.Number(
                        label="预热步数",
                        value=settings.get("wan_training", {}).get("lr_warmup_steps", 0),
                        precision=0,
                        minimum=0,
                        info="学习率逐渐增加的步数",
                        interactive=True
                    )

                gr.Markdown("#### 时间步采样")
                with gr.Row():
                    wan_adv_timestep_sampling = gr.Dropdown(
                        label="时间步采样方法",
                        choices=[
                            "sigma - SD3默认，平衡各噪声级别",
                            "uniform - 均匀随机，所有timestep概率相同",
                            "sigmoid - sigmoid变换，更关注中间噪声",
                            "shift - sigmoid+shift，可调整分布",
                            "flux_shift - FLUX优化，适合高分辨率",
                            "qwen_shift - Qwen优化策略",
                            "logsnr - 基于log-SNR，理论更优"
                        ],
                        value=settings.get("wan_training", {}).get("timestep_sampling", "sigma - SD3默认，平衡各噪声级别"),
                        info="影响训练时噪声分布",
                        interactive=True
                    )
                    wan_adv_flow_shift = gr.Number(
                        label="离散流偏移",
                        value=settings.get("wan_training", {}).get("discrete_flow_shift", 1.0),
                        minimum=0.1,
                        maximum=10.0,
                        info="Euler调度器的流偏移参数",
                        interactive=True
                    )

                wan_adv_weighting = gr.Dropdown(
                    label="权重方案",
                    choices=[
                        "none - 无权重，所有timestep权重相同",
                        "logit_normal - logit正态分布，SD3论文推荐",
                        "mode - 模式权重",
                        "cosmap - 余弦映射，平滑过渡",
                        "sigma_sqrt - sigma平方根，强调低噪声"
                    ],
                    value=settings.get("wan_training", {}).get("weighting_scheme", "none - 无权重，所有timestep权重相同"),
                    info="时间步分布的权重策略",
                    interactive=True
                )

                gr.Markdown("#### 内存优化")
                wan_adv_gradient_checkpointing = gr.Checkbox(
                    label="启用梯度检查点",
                    value=settings.get("wan_training", {}).get("enable_gradient_checkpointing", True),
                    info="降低显存使用，但会略微降低训练速度",
                    interactive=True
                )

                gr.Markdown("#### 训练控制")
                wan_adv_seed = gr.Number(
                    label="随机种子",
                    value=settings.get("wan_training", {}).get("seed", 42),
                    precision=0,
                    info="设置随机种子以获得可复现的结果",
                    interactive=True
                )

                gr.Markdown("#### 采样生成")

                with gr.Row():
                    wan_adv_sample_epochs = gr.Number(
                        label="每N轮生成样本",
                        value=settings.get("wan_training", {}).get("sample_every_n_epochs", 0),
                        precision=0,
                        minimum=0,
                        info="例如：1=每轮都生成预览，5=每5轮生成一次，0=禁用预览生成",
                        interactive=True
                    )
                    wan_adv_sample_steps = gr.Number(
                        label="采样推理步数",
                        value=settings.get("wan_training", {}).get("sample_steps", 20),
                        precision=0,
                        minimum=1,
                        maximum=1000,
                        info="全局默认推理步数，脚本默认20，Wan推荐40。提示词文件中可用 --s 覆盖此值",
                        interactive=True
                    )

                with gr.Row():
                    wan_adv_sample_solver = gr.Dropdown(
                        label="采样器算法",
                        choices=["unipc", "dpm++", "vanilla", "sa_ode_stable"],
                        value=settings.get("wan_training", {}).get("sample_solver", "unipc"),
                        info="unipc=默认推荐 | dpm++=DPM求解器 | vanilla=基础 | sa_ode_stable=稳定ODE求解器",
                        interactive=True
                    )

                wan_adv_sample_prompts = gr.Textbox(
                    label="样本提示词",
                    value=settings.get("wan_training", {}).get("sample_prompts", ""),
                    placeholder="a beautiful sunset over the ocean\na cat playing with a ball --s 30",
                    lines=5,
                    info="直接输入提示词，每行一个。可在行末使用 --s 覆盖全局采样步数",
                    interactive=True
                )

                gr.Markdown("""
                **提示词格式示例:**
                ```
                a woman walking --s 40 --w 640 --h 480 --f 16
                a cat playing
                a dog running in the park --s 40
                ```
                每行一个提示词，可选参数：--s=推理步数 --w=宽度 --h=高度 --f=帧数 --d=种子 --g=引导比例 --n=负面提示词
                """)

                gr.Markdown("#### 日志与监控")
                wan_adv_logging_dir = gr.Textbox(
                    label="TensorBoard日志目录",
                    value=settings.get("wan_training", {}).get("logging_dir", ""),
                    placeholder="./logs",
                    info="留空则不启用TensorBoard",
                    interactive=True
                )
                wan_adv_wandb_name = gr.Textbox(
                    label="WandB运行名称",
                    value=settings.get("wan_training", {}).get("wandb_run_name", ""),
                    placeholder="wan_training_run",
                    info="留空则不启用WandB",
                    interactive=True
                )

                # 添加实时训练曲线图
                gr.Markdown("### 📊 实时训练监控")
                with gr.Row():
                    with gr.Column():
                        wan_loss_plot = gr.Plot(label="Loss 曲线", value=create_loss_plot())
                    with gr.Column():
                        wan_progress_plot = gr.Plot(label="训练进度 & 学习率", value=create_progress_plot())

                # 创建定时更新函数
                def update_plots():
                    return create_loss_plot(), create_progress_plot()

                # 设置定时器更新图表（每2秒更新一次，更实时）
                wan_plot_timer = gr.Timer(2)
                wan_plot_timer.tick(fn=update_plots, outputs=[wan_loss_plot, wan_progress_plot])

                wan_train_run_btn.click(
                    fn=run_wan_training,
                    inputs=[
                        wan_train_task,
                        wan_train_dataset_file,
                        wan_train_dataset_text,
                        wan_train_dit,
                        wan_train_is_wan22,
                        wan_train_dit_low_noise,
                        wan_train_epochs,
                        wan_train_lr,
                        wan_train_network_dim,
                        wan_train_network_alpha,
                        wan_train_blocks_to_swap,
                        wan_train_fp8,
                        wan_train_output_dir,
                        wan_train_output_name,
                        wan_train_save_every,
                        wan_train_use_network_weights,
                        wan_train_network_weights_path,
                        wan_adv_attention,
                        wan_adv_mixed_precision,
                        wan_adv_optimizer,
                        wan_adv_grad_accum,
                        wan_adv_max_grad_norm,
                        wan_adv_lr_scheduler,
                        wan_adv_lr_warmup,
                        wan_adv_timestep_sampling,
                        wan_adv_flow_shift,
                        wan_adv_weighting,
                        wan_adv_gradient_checkpointing,
                        wan_adv_seed,
                        wan_adv_sample_epochs,
                        wan_adv_sample_prompts,
                        wan_adv_sample_steps,
                        wan_adv_sample_solver,
                        wan_adv_logging_dir,
                        wan_adv_wandb_name
                    ],
                    outputs=wan_train_output
                )
                wan_train_stop_btn.click(fn=stop_training, outputs=wan_train_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

