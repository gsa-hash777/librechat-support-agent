# 本地向量混合检索与重排

这个扩展让同一套业务知识支持三条可切换路径。默认仍是 `lexical`；是否改变默认值由固定测试集的证据召回、排序、延迟和内存共同决定。它不修改原始 100 条 Agent 基线，也不把检索分数当作答案正确率。

## 使用

在 VS Code 打开仓库根目录 `SupportAgent.code-workspace`，执行任务：

1. `Support: Prepare CPU retrieval models`：安装可选依赖并下载锁定版本的模型，校验 SHA256 / Git blob SHA1；下载失败不会运行损坏文件。
2. `Support: Start with hybrid retrieval options`：构建可选 CPU 镜像并重建工单服务，刷新已有 LibreChat Agent 的工具 schema，不调用生成模型。LibreChat 和工单数据库继续使用原有卷。
3. 打开 <https://localhost:8443/console/knowledge>，填写具体业务问题，在“检索方式”选择关键词、混合检索、混合检索并重排。

同一个 `searchSupportKnowledge` 工具新增可选 `mode=lexical|hybrid|rerank`。不填时使用 `SUPPORT_RETRIEVAL_MODE`，默认 `lexical`。模型未启用、文件损坏或推理失败返回明确的 503，不静默退回其他算法。

模型位于 `.local/models/support-retrieval`；只读挂载到容器 `/models`。文档向量缓存位于 Docker 数据卷 `/data/retrieval-vectors`，离线评测使用每个实验独立的目录。无需 GPU、PyTorch、外部向量数据库或 embedding API Key。部署的 768 MiB 是容器上限，不是实际空闲内存或容量保证；开发机应保留足够余量。

回到标准轻量服务：执行原有 `Support: Start` 任务，或 `docker compose -f compose.support.yml up -d --build --wait support-agent`。它不加载神经模型；模型文件和业务卷保留。

## 实现

`app/hybrid.py` 负责组织范围内的向量缓存、检索和融合。文档从已认证组织的当前版本分块读取；不安全文档先排除。缓存键包含组织、模型指纹、分窗策略、标题/小标题/正文的内容摘要，文档更新不会复用旧内容的向量。

| 路径 | 流程 |
| --- | --- |
| lexical | 现有中文词项/BM25 排序 → 原有证据判断 |
| hybrid | 词项前 10 + 归一化向量点积前 10 → 等权 RRF 融合 → 前 k → 同一证据判断 |
| rerank | 上述融合前 10 → cross-encoder 对“问题、段落”逐对打分 → 前 k → 同一证据判断 |

RRF 每路贡献 `1 / (60 + rank)`，相加后排序，不直接混加 BM25 与余弦的不同量纲。cross-encoder 的原始 logit 仅用于相对排序，不是事实置信度。证据不足或缺少问题上下文时，三条路径都运行 `app/retrieval.py` 同一套规则；这一轮没有同时修改判断规则或 Agent 提示词。

`app/hybrid_models.py` 只从本机加载经过校验的量化 ONNX 文件，CPU 两线程、batch 1。最多保留一个模型会话，切换时释放旧会话。BGE 使用 CLS 表征，多窗口先平均再 L2 归一；超长文档与查询保留溢出窗口。重排对完整窗口逐一评分后取最大值。它是低内存折中，并非最佳吞吐部署。

模型来源及固定版本见 `evaluation/retrieval_models.json`：

- [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) 的 [Xenova 量化 ONNX 转换](https://huggingface.co/Xenova/bge-small-zh-v1.5)，MIT。
- [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) 的官方 `model_quint8_avx2.onnx`，Apache 2.0。

`app/main.py` 在只读事务内校验身份并读取组织分块，释放事务后推理，返回前再次校验账号、组织、API Key 或会话是否失效，再记录审计。推理不会持有 SQLite 写锁。未经证据判断的 `ranked_results` 仅供离线评测，API 和页面不返回该字段。

当前是小语料精确向量扫描，上限 2000 个安全分块；到规模上限明确失败，需要另外设计向量索引与容量验证。串行推理锁避免本机并发模型堆积，尚未建立多用户吞吐或排队 SLA。

## 可复核的比较方法

`evaluation/hybrid_cases.json` 在首次真实比较前固定 36 条问题：12 条开发集、24 条测试集。问题由项目开发过程编写，不是生产流量或外部独立专家数据。首次测试后再次使用只能称回归测试。

主语料是从 `knowledge.json` 原文转换并实际导入的 `data/documents/knowledge.md`（7 个分块）。单独以原始 16 篇种子资料作为对照。不混入功能说明文档，也不把同一知识的两份副本当作独立证据。KD 文档中的正确原文与原 KB 证据等价，通过原文 span、摘要和位置校验判断。

- Hit@1 / Hit@3：前 1 / 3 段完整覆盖至少一个所需证据单元的问题比例。
- 主指标 Recall@3：每题“前三段覆盖的所需证据单元数 / 该题全部所需单元数”，再对可回答问题取平均。
- MRR@3：最小的前 r 段覆盖至少一个完整证据单元时记 `1/r`，否则 0；跨相邻分块可以联合覆盖。
- 缺证识别率：标准为无答案的问题中，返回 `insufficient` 的比例。模糊问题则单独检查 `clarify`。
- 原始召回 `raw` 与经过证据判断后的 `final` 分别计算。没有给生成答案打分，不能据此宣称 Agent 成功率或幻觉率改善。
- 每个路径/语料独立进程、空文档缓存启动；记录初始化和首个未评分查询的冷启动，再对每道题重复 3 次测温缓存 P50/P95。每次仍计算问题向量；重排切换会话耗时也计入。下载、HTTP 和生成回答不计入本指标。
- 峰值 RSS 是操作系统记录的该进程最高工作集；不是模型文件大小，也不是整个 Docker/LibreChat 内存。

VS Code 的 `Retrieval comparison (12 dev cases)` 和 `Retrieval regression (24 fixed test cases)` 会生成新的 `.local/hybrid-evaluation/<时间>/report.html` 与原始 JSON，保留模型/输入/代码指纹和源码快照，不覆盖旧实验。CPU 推理不调用付费生成模型。

## 首次固定测试结果（2026-11-10）

三种方案、两套语料均完成 24/24，零执行错误。主语料是 7 个分块的业务手册，以下证据指标的分母为 16 个可回答问题；开发 12 题不混入测试分母。

| 方案 | Recall@3 | Hit@1 | MRR@3 | 温缓存 P95 | 进程峰值 RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 关键词 | 78.125% | 81.25% | 0.8125 | 6.49 ms | 43.14 MiB |
| 混合检索 | 84.375% | 68.75% | 0.78125 | 14.38 ms | 132.07 MiB |
| 混合 + 重排 | 96.875% | 68.75% | 0.84375 | 2548.29 ms | 555.73 MiB |

本表 raw/final 证据覆盖相同，仍分别保留原始输出。重排多找回 4 个证据单元（13/18 → 17/18），逐题宏平均增加 18.75 个百分点；第一条命中却下降 12.5 个百分点，因此不是所有指标都改善。原始 16 篇短资料对照的 Recall@3 为 65.625% / 100% / 100%，不能把较好语料的成绩替代主结果。

三臂缺证标签一致均 1/5、明确追问均 0/3。部分否定性证据可能支持安全回答，标签未命中不等于模型幻觉；本轮没有评生成答案。随后应改进问题澄清与证据充足性判断，再扩大资料与固定测试集验证。保持关键词默认，按需启用另外两种路径。

真实工单容器峰值 658.15 MiB / 上限 768 MiB，无 OOM；这不是多人并发容量承诺。模型更新需要重启，旧向量缓存尚无自动清理，健康检查只验证业务服务而非模型已加载，串行推理尚未配置队列超时。

完整验收见 [VALIDATION.md](VALIDATION.md) 与工作区 `outputs/LibreChat-混合检索与重排对比.html`。开发评分器曾把公开投影省略 org_id 误判为来源错误，保留原数据并离线修正后才冻结最终测试；正式测试没有事后改规则。旧 Agent 95/100 基线和已有知识库实验保留原口径。
