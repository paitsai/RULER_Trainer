# datamaker

RULER 风格的"人造数据 → 提取有用信息 → 解答最终答案"三段式训练数据流水线。

## 设计

对标 `RULER/scripts/data/` 的合成思路（噪音文本 + 隐藏信息 + 受控长度），但为 agent
训练输出显式的中间步骤：

1. **人造数据 (synthesize)**：`BaseTask.generate_input_output()` 生成一段受污染的
   长文本（haystack）并在其中按随机位置藏入"有用信息"（needle），同时生成 query。
2. **提取有用信息 (extract)**：按 query 定位到相关 needle，得到 gold 的
   `extraction` 字段（模型需要先输出的中间步骤）。
3. **解答最终答案 (answer)**：从提取出的信息中推导出 `outputs` 字段（最终答案）。

每条样本输出为原始 JSONL 字段：

```
{
  "index": 0,
  "task": "niah",
  "input": "...",            // 完整 prompt（含 model chat template）
  "extraction": "...",       // 应从文本中提取的有用信息（中间步骤 gold）
  "outputs": ["..."],        // 基于提取信息得到的最终答案
  "length": 1013,            // input token 数 + tokens_to_generate
  "answer_prefix": "...",    // 接在 input 末尾、用于引导模型作答的前缀
  "token_position_answer": 49,   // 答案在 input 中的 token 位置（可做深度分析）
  "meta": "{...}"            // query、num_haystack、needle 配置等
}
```

长度控制与 RULER 一致：用真实 tokenizer（`--tokenizer_path` + `--tokenizer_type hf`）
按 "input tokens + tokens_to_generate <= max_seq_length" 二分搜索最优 haystack 大小；
不传 tokenizer 时退化为空白切分（默认 `none`）。

## 使用

```bash
# 单个任务配置
python -m datamaker.prepare \
    --save_dir ./output \
    --config datamaker/configs/niah.yaml \
    --task niah_single_1 \
    --max_seq_length 4096 \
    --num_samples 500 \
    --tokenizer_path gpt2 \
    --tokenizer_type hf

# 不使用 HF tokenizer（空白切分近似）
python -m datamaker.prepare \
    --save_dir ./output \
    --config datamaker/configs/niah.yaml \
    --task niah_multikey_1 \
    --max_seq_length 4096 \
    --num_samples 10

# 用 llama3 类 chat template
python -m datamaker.prepare \
    --config datamaker/configs/niah.yaml \
    --task niah_single_1 \
    --max_seq_length 4096 \
    --model_template_type llama3 \
    ...
```

输出到 `{save_dir}/{task_name}/train.jsonl`。

## 任务与扩展

当前已实现 4 类任务（对标 RULER 的分类），复杂度参数在 `datamaker/configs/tasks.yaml` 中配置。

### 1. NIAH — Retrieval（`datamaker/tasks/niah.py`）
长文本中藏 key-value 针，按 query 提取相关针再回答 value。

| 参数 | 含义 |
|---|---|
| `type_haystack` | `noise`（随机噪音句）/ `needle`（干扰针）/ `essay`（Paul Graham 文章，需 RULER 数据） |
| `type_needle_k` / `type_needle_v` | key/value 类型：`words` / `numbers` / `uuids` |
| `num_needle_k` / `num_needle_v` / `num_needle_q` | 隐藏 key 数 / 每 key 的 value 数 / query 查的 key 数 |

### 2. Variable Tracking — Multi-hop Tracing（`variable_tracking.py`）
噪音文本中混入变量赋值链 `VAR A = 70263; VAR B = VAR A; ...`，按 value 反查被赋值的全部变量。

| 参数 | 含义 |
|---|---|
| `num_chains` | 链数量 |
| `num_hops` | 每条链的跳数（越多越难） |

### 3. Common Words Extraction — Aggregation（`common_words_extraction.py`）
生成编号词表，部分词高频重复（`freq_cw` 次）、其余低频（`freq_ucw` 次），提取词频后回答 top 高频词。词表来自 wonderwords（与 RULER 一致）。

| 参数 | 含义 |
|---|---|
| `num_cw` | 需要找出的高频词个数 |
| `freq_cw` / `freq_ucw` | 高频 / 低频词重复次数 |

### 4. Freq Words Extraction — Aggregation（`freq_words_extraction.py`）
随机 6 字母"密文词"词表，按 Zipf 分布（`alpha`）抽样词频，排名第 1 的词替换为 `...` 噪音，提取词频后回答 top 3 词（无需 scipy，内置 zeta 数值近似）。

| 参数 | 含义 |
|---|---|
| `alpha` | Zipf 分布参数，越小分布越均匀越难 |

### 5. Numeric Aggregation — Aggregation（`numeric_aggregation.py`）
噪音句中散布 `Special number XX` 标记数字，提取所有标记数字后求和。

| 参数 | 含义 |
|---|---|
| `num_items` | 标记数字个数 |
| `digits` | 数字位数 |

### 6. Temporal Ordering — Multi-hop（`temporal_ordering.py`）
噪音句中混入带时间戳事件 `[YYYY-MM-DD HH:MM] 事件`，提取并排序后回答最早事件。

| 参数 | 含义 |
|---|---|
| `num_events` | 事件个数 |

### 7. Entity Counting — Aggregation（`entity_counting.py`）
噪音句中散布多种动物出现记录，提取目标动物出现次数后计数。

| 参数 | 含义 |
|---|---|
| `target_freq` | 目标动物出现次数（答案） |
| `num_distractors` | 干扰动物种类数 |

### 8. Multi-hop Fact Chain — Multi-hop（`multi_hop_fact.py`）
人物关系链 `Henry is the colleague of Jack. Jack is the colleague of Frank.` + 居住城市事实，多跳推理回答 `Where does the colleague of the colleague of Henry live?`。

| 参数 | 含义 |
|---|---|
| `num_hops` | 关系链跳数 |
| `num_distractors` | 干扰人物/城市对个数 |

### 9. QA（`qa.py`）
真实数据（SQuAD dev / HotpotQA dev distractor），取真实 question + 支持文档，用其他文档填充长度（`--data_dir` 指定数据目录，内含 `squad.json` / `hotpotqa.json`，文件放 `datamaker/data/`）。

| 参数 | 含义 |
|---|---|
| `dataset` | `squad` 或 `hotpotqa` |

所有任务的配置示例见 `datamaker/configs/tasks.yaml`（niah ×6、vt ×3、cwe ×2、fwe ×2、num_agg ×2、temporal ×2、entity_count ×2、fact_hop ×2、qa ×2）。

新增任务：在 `datamaker/tasks/` 下继承 `BaseTask`，实现
`generate_input_output(num_source, index)`（返回 context / extraction / answers / meta），
并在 `datamaker/tasks/__init__.py` 的 `TASK_REGISTRY` 注册，即可复用
二分长度控制、answer_prefix 切分、JSONL 输出等逻辑。

## 验证

```bash
python -m datamaker.prepare --save_dir /tmp/out --config datamaker/configs/tasks.yaml \
    --task vt_1 --max_seq_length 1024 --num_samples 3
head -1 /tmp/out/vt_1/train.jsonl | python -m json.tool
```