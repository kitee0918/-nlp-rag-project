"""测试数据加载"""

import os

# 设置 HuggingFace 镜像（中国大陆用户）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("开始测试...")
print("使用镜像站点:", os.environ.get('HF_ENDPOINT', 'default'))

# 测试导入
try:
    from datasets import load_dataset
    print("[OK] datasets 导入成功")
except ImportError as e:
    print(f"[FAIL] datasets 导入失败: {e}")
    exit(1)

# 测试数据加载
print("\n正在加载 HotpotQA 数据集 (小样本)...")
try:
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    print("[OK] 数据集加载成功，共", len(dataset), "条数据")
    
    # 显示第一条数据
    first_item = dataset[0]
    print(f"\n第一条数据示例:")
    print(f"  可用字段: {list(first_item.keys())}")
    print(f"  问题: {first_item['question']}")
    print(f"  答案: {first_item['answer']}")
    print(f"  类型: {first_item.get('type', 'N/A')}")
    print(f"  难度: {first_item.get('level', 'N/A')}")
    
    # 显示 context 结构
    context = first_item['context']
    print(f"\n  context 类型: {type(context)}")
    print(f"  context 键: {list(context.keys()) if isinstance(context, dict) else 'N/A'}")
    
    if isinstance(context, dict):
        titles = context.get('title', [])
        sentences = context.get('sentences', [])
        print(f"  文档数量: {len(titles)}")
        print(f"  文档标题: {titles}")
        if sentences:
            print(f"  第一个文档句子数: {len(sentences[0])}")
            print(f"  第一个文档内容: {sentences[0][:2]}")
    
    # 显示 supporting_facts
    sf = first_item['supporting_facts']
    print(f"\n  supporting_facts 类型: {type(sf)}")
    print(f"  supporting_facts 键: {list(sf.keys()) if isinstance(sf, dict) else 'N/A'}")
    if isinstance(sf, dict):
        print(f"  支持事实标题: {sf.get('title', [])}")
        print(f"  支持事实句子ID: {sf.get('sent_id', [])}")
    
except Exception as e:
    print("[FAIL] 数据加载失败:", e)
    import traceback
    traceback.print_exc()

print("\n测试完成!")
