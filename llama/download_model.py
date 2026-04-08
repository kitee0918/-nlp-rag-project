"""
HuggingFace CLI 模型下载脚本 (Colab 兼容版)
参考文档: https://huggingface.co/docs/huggingface_hub/en/guides/cli

Colab 使用方法:
1. 上传此脚本到 Colab 或直接复制代码
2. 运行第一个单元格安装依赖
3. 运行下载代码
"""

import os
import subprocess
import sys
from pathlib import Path

# ============================================================
# Colab 环境检测与配置
# ============================================================

IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
    print("✓ 检测到 Colab 环境")
except ImportError:
    print("✓ 本地环境运行")


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
MODEL_REPO_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# 要下载的特定文件 (可选，留空则下载整个仓库)
# 例如: "config.json", "pytorch_model.bin", "tokenizer.json"
SPECIFIC_FILE = None  # 例如: "config.json"

# 本地保存目录
# Colab 建议保存到 /content 或 Google Drive
if IN_COLAB:
    LOCAL_DIR = "/content/models/llama-3.1-8b"  # Colab 临时存储
    # LOCAL_DIR = "/content/drive/MyDrive/models/llama-3.1-8b"  # Google Drive (需先挂载)
else:
    LOCAL_DIR = None  # 本地使用默认缓存目录

# HuggingFace Token
# 方式1: 直接设置 (不推荐在生产环境使用)
HF_TOKEN = "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"

# 方式2: Colab Secrets (推荐) - 取消下面的注释使用
# from google.colab import userdata
# HF_TOKEN = userdata.get('HF_TOKEN')

# 方式3: 环境变量
# HF_TOKEN = os.environ.get("HF_TOKEN")

# ============================================================


def main():
    print("=" * 60)
    print("HuggingFace 模型下载工具 (Colab 兼容版)")
    print("=" * 60)
    
    # Colab 特定设置
    if IN_COLAB:
        print("\n[Colab] 安装依赖...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub", "-q"])
    
    # 步骤1: 检查并安装依赖
    if not check_huggingface_hub():
        install_huggingface_hub()
    
    # 步骤2: 登录 (Llama 等受限模型需要授权)
    login_huggingface(HF_TOKEN)
    
    # 步骤3: 创建保存目录
    if LOCAL_DIR:
        Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)
        print(f"保存目录: {LOCAL_DIR}")
    
    # 步骤4: 下载模型
    print(f"\n准备下载模型: {MODEL_REPO_ID}")
    if SPECIFIC_FILE:
        print(f"目标文件: {SPECIFIC_FILE}")
    
    # 使用 Python API 下载 (推荐)
    download_model_python(
        repo_id=MODEL_REPO_ID,
        filename=SPECIFIC_FILE,
        local_dir=LOCAL_DIR,
        token=HF_TOKEN
    )
    
    print("\n" + "=" * 60)
    print("下载完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
