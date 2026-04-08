"""
HuggingFace CLI 模型下载脚本
参考文档: https://huggingface.co/docs/huggingface_hub/en/guides/cli

使用方法:
1. 首先安装 huggingface_hub: pip install -U huggingface_hub
2. 登录 HuggingFace: hf auth login 或设置 HF_TOKEN 环境变量
3. 运行此脚本下载模型
"""

import os
import subprocess
import sys
from pathlib import Path


def check_huggingface_hub():
    """检查是否安装了 huggingface_hub"""
    try:
        import huggingface_hub
        print(f"✓ huggingface_hub 已安装，版本: {huggingface_hub.__version__}")
        return True
    except ImportError:
        print("✗ huggingface_hub 未安装")
        return False


def install_huggingface_hub():
    """安装 huggingface_hub"""
    print("正在安装 huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])
    print("✓ 安装完成")


def login_huggingface(token=None):
    """
    登录 HuggingFace
    
    参数:
        token: HuggingFace Access Token (可选)
               可从 https://huggingface.co/settings/tokens 获取
    """
    from huggingface_hub import login
    
    if token:
        login(token=token)
        print("✓ 登录成功")
    else:
        # 如果没有提供 token，会提示输入
        login()
        print("✓ 登录成功")


def download_model_cli(repo_id: str, filename: str = None, local_dir: str = None):
    """
    使用 CLI 命令下载模型
    
    参数:
        repo_id: 模型仓库ID，如 "meta-llama/Llama-2-7b-hf"
        filename: 要下载的文件名 (可选，不指定则下载整个仓库)
        local_dir: 本地保存目录 (可选)
    """
    cmd = ["hf", "download", repo_id]
    
    if filename:
        cmd.append(filename)
    
    if local_dir:
        cmd.extend(["--local-dir", local_dir])
    
    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ 下载成功")
        print(result.stdout)
    else:
        print(f"✗ 下载失败: {result.stderr}")


def download_model_python(repo_id: str, filename: str = None, local_dir: str = None, 
                          token: str = None):
    """
    使用 Python API 下载模型
    
    参数:
        repo_id: 模型仓库ID，如 "meta-llama/Llama-2-7b-hf"
        filename: 要下载的文件名 (可选，不指定则下载整个仓库)
        local_dir: 本地保存目录 (可选)
        token: HuggingFace token (可选)
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    
    if filename:
        # 下载单个文件
        print(f"正在下载文件: {repo_id}/{filename}")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            token=token
        )
    else:
        # 下载整个仓库
        print(f"正在下载模型仓库: {repo_id}")
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=token
        )
    
    print(f"✓ 下载完成，保存路径: {path}")
    return path


# ============================================================
# 配置区域 - 在这里修改你要下载的模型
# ============================================================

# 模型仓库ID (从 HuggingFace 模型页面获取)
# 例如: "meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1"
MODEL_REPO_ID = "meta-llama/Llama-2-7b-hf"

# 要下载的特定文件 (可选，留空则下载整个仓库)
# 例如: "config.json", "pytorch_model.bin", "tokenizer.json"
SPECIFIC_FILE = None  # 例如: "config.json"

# 本地保存目录 (可选，留空则使用默认缓存目录)
LOCAL_DIR = None  # 例如: "./models/llama-2-7b"

# HuggingFace Token (可选，也可以通过环境变量 HF_TOKEN 设置)
HF_TOKEN = None  # 或 os.environ.get("HF_TOKEN")

# ============================================================


def main():
    print("=" * 60)
    print("HuggingFace 模型下载工具")
    print("=" * 60)
    
    # 步骤1: 检查并安装依赖
    if not check_huggingface_hub():
        install_huggingface_hub()
    
    # 步骤2: 登录 (如果需要私有模型或有限制的模型)
    # 对于需要授权的模型(如Llama)，需要先登录
    # 取消下面这行的注释来登录:
    # login_huggingface(HF_TOKEN)
    
    # 步骤3: 下载模型
    print(f"\n准备下载模型: {MODEL_REPO_ID}")
    if SPECIFIC_FILE:
        print(f"目标文件: {SPECIFIC_FILE}")
    if LOCAL_DIR:
        print(f"保存目录: {LOCAL_DIR}")
    
    # 使用 Python API 下载 (推荐)
    download_model_python(
        repo_id=MODEL_REPO_ID,
        filename=SPECIFIC_FILE,
        local_dir=LOCAL_DIR,
        token=HF_TOKEN
    )
    
    # 或者使用 CLI 命令下载:
    # download_model_cli(
    #     repo_id=MODEL_REPO_ID,
    #     filename=SPECIFIC_FILE,
    #     local_dir=LOCAL_DIR
    # )


if __name__ == "__main__":
    main()
