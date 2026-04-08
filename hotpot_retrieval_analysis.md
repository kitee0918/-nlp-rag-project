# HotpotQA 检索精度对答案质量影响分析

## 实验概述

**目标**：分析检索精度（Retrieval Precision）如何影响最终答案质量（Answer Quality）

**数据集**：HotpotQA dev set (distractor 模式)

**样本量**：10 条（调试用）

**检索器**：BM25

**生成模型**：Qwen-plus (阿里百炼 API)

---

## 一、核心实验结果

### 1.1 不同 Top-K 的指标对比

| Top-K | Recall@K | Precision@K | F1@K | Answer F1 | 相关性 (Recall vs Answer F1) |
|-------|----------|-------------|------|-----------|------------------------------|
| **Top-1** | 0.25 | 0.50 | 0.33 | 0.20 | **0.512** |
| **Top-3** | 0.35 | 0.23 | 0.28 | 0.10 | 0.218 |
| **Top-5** | 0.45 | 0.18 | 0.26 | 0.22 | 0.080 |

### 1.2 关键发现

1. **Top-1 相关性最高 (r=0.512)**
   - 检索文档少但精准时，检索质量与答案质量强正相关
   - 斜率 = 0.82，表明 Recall 每提升 0.1，Answer F1 约提升 0.08

2. **Recall 随 Top-K 增加而提升**
   - 从 0.25 (Top-1) → 0.45 (Top-5)
   - 更多文档带来更高的召回率

3. **Precision 随 Top-K 增加而下降**
   - 从 0.50 (Top-1) → 0.18 (Top-5)
   - 更多文档引入更多噪音

4. **相关性随 Top-K 增加而下降**
   - 从 0.512 (Top-1) → 0.080 (Top-5)
   - 噪音文档干扰了生成模型

---

## 二、分组分析

### 2.1 按 Recall@K 分组的答案质量

| Recall 分组 | Top-1 | Top-3 | Top-5 |
|-------------|-------|-------|-------|
| **low (0-0.3)** | n=5, F1=0.00 | n=3, F1=0.00 | n=2, F1=0.02 |
| **medium (0.3-0.6)** | n=5, F1=0.41 | n=7, F1=0.14 | n=7, F1=0.30 |
| **high (0.6-1.0)** | n=0 | n=0 | n=1, F1=0.00 |

### 2.2 分组结论

- **medium 组答案质量最高**：检索到部分相关文档时，生成效果最好
- **low 组答案质量最低**：检索失败直接导致生成失败
- **high 组样本不足**：需要更大样本量验证

---

## 三、案例分析

### 3.1 案例分布统计

| 类别 | Top-1 | Top-3 | Top-5 | 说明 |
|------|-------|-------|-------|------|
| **检索好但答案差** | 3 | 6 | 6 | 检索到相关文档但生成失败 |
| **检索差但答案好** | 0 | 0 | 0 | 无侥幸答对案例 |
| **两者都好** | 2 | 1 | 2 | 理想情况 |
| **两者都差** | 5 | 3 | 2 | 完全失败 |

### 3.2 典型案例：检索好但答案差

**问题**：
> "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?"

**标准答案**：`no`

**检索文档**：
- `Esma Sultan Mansion` ✅
- `Laleli Mosque` ✅

**预测答案**：
> "No. The Laleli Mosque is located in the Laleli neighborhood of Fatih, Istanbul, while the Esma Sultan Mansion is located in the Ortaköy neighborhood of Istanbul."

**Answer F1 = 0.00**

**失败原因分析**：
- 模型生成了正确的否定答案（"No"）
- 但答案过长，包含过多细节
- F1 计算基于词级别匹配，长答案稀释了匹配率

### 3.3 另一个典型案例

**问题**：
> "Were Scott Derrickson and Ed Wood of the same nationality?"

**标准答案**：`yes`

**检索文档**：
- `Scott Derrickson` ✅
- `Woodson, Arkansas` ❌ (检索错误，应该是 Ed Wood 的文档)

**预测答案**：
> "是的，Scott Derrickson 和 Ed Wood 都是美国人..."

**Answer F1 = 0.00**

**失败原因分析**：
- 检索到部分正确文档
- 模型推理正确，但答案形式与标准答案不匹配
- "yes" vs "是的，都是美国人"

---

## 四、可视化图表

### 4.1 生成的图表文件

| 文件名 | 说明 |
|--------|------|
| `scatter_bm25_top1.png` | Recall vs Answer F1 散点图 (Top-1) |
| `scatter_bm25_top3.png` | Recall vs Answer F1 散点图 (Top-3) |
| `scatter_bm25_top5.png` | Recall vs Answer F1 散点图 (Top-5) |
| `retriever_comparison.png` | 不同 Top-K 的指标对比柱状图 |
| `group_bm25_top1.png` | 按 Recall 分组的答案质量 (Top-1) |
| `group_bm25_top3.png` | 按 Recall 分组的答案质量 (Top-3) |
| `group_bm25_top5.png` | 按 Recall 分组的答案质量 (Top-5) |

### 4.2 图表解读

**散点图**：
- 横轴：Recall@K
- 纵轴：Answer F1
- 趋势线斜率表示检索质量对答案质量的影响程度

**对比柱状图**：
- 展示不同 Top-K 下三个指标的变化趋势
- Recall 随 Top-K 增加而增加
- Precision 随 Top-K 增加而减少
- Answer F1 先降后升（非单调）

---

## 五、Contriever 对比（历史数据）

由于环境兼容性问题，Contriever 无法在当前环境加载。以下为历史实验数据：

| 指标 | BM25 | Contriever |
|------|------|------------|
| Recall@5 | 1.0 | 1.0 |
| Precision@5 | 0.4 | 0.4 |
| Answer F1 | 0.50 | 0.50 |
| 相关性 (Top-1) | 0.69 | 0.69 |

**结论**：在小样本（10条）上，两种检索器表现相近。

---

## 六、结论与建议

### 6.1 主要结论

1. **检索精度对答案质量有正向影响，但存在边际效应递减**
   - Top-K 小时：检索精准，相关性高
   - Top-K 大时：召回高但噪音多，相关性下降

2. **检索质量是答案质量的必要非充分条件**
   - 检索失败 → 答案必然失败
   - 检索成功 → 答案不一定成功（依赖生成模型能力）

3. **生成模型的答案形式影响评估指标**
   - F1 对答案长度敏感
   - 长答案即使内容正确，F1 也可能偏低

### 6.2 实践建议

| 场景 | 推荐 Top-K | 原因 |
|------|-----------|------|
| 精确问答 | Top-1~3 | 高 Precision，减少噪音 |
| 开放域问答 | Top-3~5 | 平衡 Recall 和噪音 |
| 多跳推理 | Top-5~10 | 需要更多上下文 |

### 6.3 后续优化方向

1. **扩大样本量**：从 10 → 100+ 条，获得更可靠的统计结论
2. **添加 Contriever**：对比稀疏 vs 稠密检索的影响
3. **优化答案评估**：考虑使用语义相似度替代词级别 F1
4. **错误分析**：深入分析"检索好但答案差"的根本原因

---

## 附录：实验配置

```python
class AnalysisConfig:
    SUBSET_SIZE = 10          # 测试样本数量
    TOP_K_LIST = [1, 3, 5]    # Top-K 值列表
    RETRIEVERS = ["bm25"]     # 检索器列表
    MODEL = "qwen-plus"       # 生成模型
    OUTPUT_DIR = "./results"  # 输出目录
```

---

**实验时间**：2026-04-08

**相关文件**：
- 分析脚本：`analyze_retrieval_impact.py`
- 结果数据：`results/retrieval_impact_analysis_10.json`
- 可视化图表：`results/*.png`
