# 工单 Agent 的测试与可复现评测

本目录提供两种独立评测。它们使用模拟业务数据，但只有第二种会调用真实模型并产生费用；测试通过数、检索命中率与 Agent 任务通过率不能混算。

| 入口 | 验证内容 | 是否调用模型 | 报告位置（相对仓库根目录） |
| --- | --- | --- | --- |
| `run.py` | 服务及集成脚本测试、40 条查询的检索基线 | 否 | `extensions/support-agent/evaluation/results/` |
| `setup.py` + `agent.py --run` | LibreChat 中真实 Agent 的回答、工具调用、数据库变化与用量 | 准备阶段不调用；`--run` 调用 | `.local/evaluation/results/<run-id>/` |

## 运行真实 Agent 评测

先按项目主说明完成安装、配置模型凭据并启动 LibreChat。以下命令均在 `E:\LibreChat` 执行，使用项目的 `.venv`；Docker Desktop 与现有 LibreChat 服务需要可用。

```powershell
.\.venv\Scripts\python.exe extensions\support-agent\evaluation\setup.py
.\.venv\Scripts\python.exe extensions\support-agent\evaluation\agent.py --dataset test
```

`setup.py` 启动独立评测服务、准备专用账号与 7 个私有 Agent，不调用模型。`agent.py` 默认只显示运行计划、题集摘要和零模型调用；只有显式添加 `--run` 才执行付费请求。

建议先运行开发集的 3 题冒烟，再视需要运行完整开发集，最后执行冻结测试集：

```powershell
.\.venv\Scripts\python.exe extensions\support-agent\evaluation\agent.py --dataset dev --limit 3 --max-budget-cny 0.5 --run
.\.venv\Scripts\python.exe extensions\support-agent\evaluation\agent.py --dataset dev --max-budget-cny 5 --run
.\.venv\Scripts\python.exe extensions\support-agent\evaluation\agent.py --dataset test --max-budget-cny 5 --run
```

这些是三次独立运行，每次都有自己的费用与报告。无需为了得到冻结集成绩反复运行开发集。`--limit` 只选取前 N 个任务；子集成绩不能写成完整 100 题成绩。

在 VS Code 打开仓库根目录的 `SupportAgent.code-workspace`，使用“终端 → 运行任务”可选择：

| 任务名称 | 实际操作 |
| --- | --- |
| `Support: Prepare isolated evaluation` | 执行 `setup.py`；不调用模型 |
| `Support: Agent smoke (3 paid tasks)` | 开发集前 3 题，0.5 元停止阈值，真实模型调用 |
| `Support: Agent baseline (100 paid tasks)` | 冻结集 100 题，5 元停止阈值，真实模型调用 |
| `Support: Stop evaluation service` | 停止 `support-eval` 容器，保留评测数据与报告 |

这里的 Agent baseline 是当前完整 Agent 方案的首轮基准，与后文检索算法的词项重合基线是不同实验。普通的 `Support: Evaluate` 仍然只执行阶段 1 的 `run.py`。

## 隔离环境与人工确认边界

评测通过 `compose.support.yml` 和 `compose.evaluation.yml` 的 `evaluation` profile 启动 `support-eval`。浏览器地址为 `http://localhost:8091`，容器内工具地址为 `http://support-eval:8090`；日常工单服务仍使用主机端口 8090。评测使用独立数据库卷 `support-eval-data` 和 `.local/evaluation/users.json`，不会把夹具写入日常工单数据库。

评测复用本机 LibreChat 平台，但使用专用登录账号及 7 个私有 Agent：`alice`、`bob`、`support1`、`support2`、`admin`、`eve`、`other_admin`。每个 Agent 绑定一个固定的评测服务身份；这验证角色与组织权限边界，不代表已实现 LibreChat 用户与工单用户之间的统一登录映射。凭据保存在 `.local/evaluation/`，不要将该目录作为公开测试数据提交。

当前 OpenAPI 暴露 **11 个工具**，包含检索、草稿、工单查询、人员查询、历史及变更提案。浏览器确认接口不在工具清单内。模型只能创建待确认草稿或变更提案，不能自动提交正式工单、改变工单状态或替用户审批。

每题开始前，评测器在隔离容器内初始化指定业务状态，并获取题前快照；完成后检查实际数据库变化。已指派、处理中、已解决等状态由测试夹具预置，**不算模型完成这些流转的证据**。评测不替模型点击确认，也不把预置工单或历史记录记为模型执行成果。完整人工确认与状态流转另由服务 HTTP 测试验证。

## 题集与成绩含义

`agent_dev_cases.json` 有 12 条开发任务，供提示词和流程调试；`agent_cases.json` 有 100 条冻结回归任务。冻结集包括知识检索 20、信息缺失 10、无答案 10、创建草稿 10、幂等 10、权限 15、生命周期 15、提示注入 10。10 条幂等任务各含两轮，因此完整冻结集计划提交 **110 条用户消息**；工具执行可能触发更多模型轮次。开发集完整运行是 12 个任务、13 条用户消息。

题集为开发代理编写的中文合成业务场景，尚未完成独立人工语义审核，不是企业生产数据或独立盲测。每题新建对话，只有同题的多轮消息共享对话。开发集和冻结集分开统计。题目、夹具与字段契约详见 [CASES.md](CASES.md)。

报告区分三类自动检查：

- **契约规则（contract）**：真实工具调用及结果、引用文档证据、草稿/提案字段、幂等与权限、数据库状态约束。
- **文本启发式（heuristic）**：预设短语是否出现；它不能证明回答在语义上正确。
- **严格规则（strict_rule）**：组合自动检查的结果，也不能直接命名为“语义准确率”或“真实业务成功率”。

人工语义复核状态保留为待复核。权限题允许模型在调用工具之前正确拒绝，但拒绝理由是否准确仍需人工查看；知识缺失题也要人工检查是否编造政策。回答中出现文档 ID，必须同时有真实检索结果支撑，不能仅靠输出一个编号通过证据检查。

报告保留计划数、已尝试数、完成数、执行失败数和未尝试数，分别展示相应分母的通过率。失败任务不会被从分母中删去；提前停止时不能将已完成子集的比例写成全部任务成绩。软件单元测试、题集结构测试与 Agent 任务运行的数量应分别报告。

## 耗时、token 与费用

真实 Agent 任务的计时从任务提交到取证完成，包含同题多轮模型调用、工具执行及最终消息和数据库状态读取；不含夹具初始化、服务部署和人工确认。当前单并发运行，P50/P95 使用所有已尝试且有耗时的任务，包含失败；缺失耗时另计。P95 使用 nearest-rank。该本机实验不能证明生产吞吐或并发承载能力。

一次任务可能触发多个模型轮次。用量优先汇总 SSE 的 `on_token_usage` 事件，按调用身份去除事件重传，不能仅取最终回答的 token。没有 SSE 用量，或保存的消息汇总明确覆盖了更多用量时，回退到持久化汇总并记录原因；两份数据只取其一，不相加。如果持久化数值小于已经观测到的 SSE 数值，则保留已知小计、将总量标为未知并停止继续计费。`tokenCount` 不是完整请求用量的替代品。

首轮 100 题实测中，恢复流可能未给解析器提供早期的实时用量事件，持久化汇总因此补齐了部分轮次。K01 的只读核对显示两次模型调用输入 2144+3278=5422、输出 119+349=468，与报告一致。回退记录的缓存明细保持区间，可观测模型调用次数不足时为 null，不能把 110 条用户消息或 usage 记录数当作模型总调用次数。后续可改进恢复流中的累计用量解析，再以新版本评测，原始基线不变。

输入 token 按供应商返回口径记录，缓存命中、未命中分别使用可观测字段；缺失的缓存分项保持未知或使用价格区间估算，不默认全命中或零成本。独立的标题生成任务、异常退出且未返回用量的调用可能不在已捕获记录内，因此记录的调用数与 token 不应宣称覆盖供应商全部计费。

费用根据 [prices.json](prices.json) 的价格快照、观测用量及缓存信息估算，保留区间和保守预算值；**这是价格估算，不是服务商账单**。缺少用量或无法给模型匹配价格时标记未知；已知费用小计不能冒充完整总费用。未来复现实验应先核对价格快照，不能在同一次冻结运行中途修改它。

`--max-budget-cny` 默认为 5。运行器在开始下一任务前检查累计保守估算：达到阈值、费用未知或前一请求是否结束不明确时停止继续提交。**5 元是停止阈值，不是供应商硬性扣费上限**；正在执行的单个任务（包括其多轮调用）可能使总额越过阈值。3 题 VS Code 冒烟任务使用同样机制，阈值为 0.5 元。

执行错误会停止本轮，模型请求不会自动重试。仅有契约或启发式规则失败、且执行和用量已完整记录的任务，可以保留失败结果后继续下一题。

## 报告、冻结与恢复

每次真实运行创建 `.local/evaluation/results/<run-id>/`，包含 `manifest.json`、逐题 JSON、`results.json`、`report.json` 与可浏览的 `report.html`。`.local/evaluation/latest.json` 指向最近一次运行目录。报告保留回答、工具调用、数据库前后快照、判定、耗时、用量与对应 LibreChat 对话链接，以便逐条复核。

运行清单保存题集、代码、价格、Agent 指令、工具 schema、知识语料与平台配置的 SHA256，以及任务 ID、模型、参数、递归限制、超时、预算和并发数。冻结集 v1 的文件 SHA256 为：

```text
befcb7171e7d9feb7b2d5766fc750a043ff5a51f6be73055caa48e9921c9830b
```

运行中发现代码或题集摘要改变时停止；恢复还要求题集、代码、价格、指令、schema、知识语料、平台配置、任务选择、模型、参数、递归限制、超时和预算与原清单一致。不要在观察测试答案后删除失败题、放宽预期或悄悄替换成绩。方案修订应创建独立运行，保留旧结果并说明差异；冻结并不构成盲测。

`--resume` 接受已有运行目录。不传 `--max-budget-cny` 时沿用原清单预算，显式传入也必须保持一致。下面的目录名是占位，须换成实际路径，并保持原来的 `--dataset`、`--limit` 选择、超时与预算配置：

```powershell
.\.venv\Scripts\python.exe extensions\support-agent\evaluation\agent.py --dataset test --max-budget-cny 5 --resume .local\evaluation\results\实际运行目录名 --run
```

恢复会保留已记录任务（包括失败任务）并跳过它们，累计费用也继续计入；它不会重跑失败题以提高成绩。若留下尚未完成取证的 `pending_case`，运行器拒绝自动恢复；应先核查原请求和费用，必要时另建运行并保留原始失败记录。用量未知、请求未结束等停止条件也不会因使用 `--resume` 而解除。

## 逐题复核与修正回归

新增的参考要求、人工复核口径、权限边界和 32 题针对性回归见 [REVIEW.md](REVIEW.md)。`--dataset repair` 运行已明确标注为后验开发集合的新任务；不修改冻结 v1 题集和旧报告。`review.py` 可生成离线人工复核页面，并验证署名决定文件后另存独立记录。参考答案和报告哈希、完整题目集合、每题理由、标准/规则修订意见均会校验，程序不能认证填写者身份。

## 阶段 1：服务测试与检索基线

以下流程不调用模型。知识与账号均为模拟数据，使用独立临时 SQLite，不读取业务凭据或改写日常运行数据：

```powershell
.\.venv\Scripts\python.exe -m pip install -r extensions\support-agent\tests\requirements.txt
.\.venv\Scripts\python.exe extensions\support-agent\evaluation\run.py
```

默认先运行 `tests/` 的服务测试与 `integration/test_*.py` 的集成脚本测试，再运行检索评测。报告分别记录两组测试成绩和逐条结果，写入本目录的 `results/report.json` 与 `results/report.md`；临时账号和数据库在 `evaluation/work` 的独立目录中创建，结束后清理。

仅执行软件测试或仅重跑检索：

```powershell
.\.venv\Scripts\python.exe -m pytest extensions\support-agent\tests extensions\support-agent\integration -q
.\.venv\Scripts\python.exe extensions\support-agent\evaluation\run.py --skip-tests --k 3 --repeats 10 --output extensions\support-agent\evaluation\results-k3
```

`--skip-tests` 的报告明确标记测试未运行，不能自动沿用此前通过成绩。服务测试检查真实数据库状态，覆盖身份绑定、角色与组织隔离、CSRF/Origin、并发幂等、取消和过期、版本冲突、降权后的确认、状态流转、历史记录、浏览器转义和持久化；OpenAPI 只提供 11 个模型工具，不暴露浏览器审批接口。

### 检索查询与指标

语料为 `data/knowledge.json` 的 16 份模拟产品手册。旧 `cases.json` 有 32 条单文档、4 条多文档、4 条无答案查询，是小规模手工功能冒烟集，没有独立盲测隔离，不能代表企业真实问题分布。它与 100 条真实 Agent 回归任务分开统计。

默认 K=3，每条查询重复 5 次，共 200 次计时请求；预热 5 条查询，顺序随机且固定，单并发。检索是确定性的，质量指标按 40 条独立查询计算，重复请求仅扩大耗时样本；脚本保留每次状态码、耗时并检查排序一致性。

- **Hit@K**：有答案查询中，前 K 份结果至少命中一份标准文档的比例。
- **Macro Recall@K**：对每个有答案查询计算前 K 份结果覆盖标准文档的比例，再取查询平均。多文档问题需覆盖全部所需证据，才能得到该题 100% Recall。
- **无证据用例空结果比例**：无答案查询中返回空列表的比例。它不是模型拒答正确率；召回相关文档也不代表文档包含答案。
- **P50/P95**：进程内 TestClient 请求耗时，包含鉴权、SQLite、检索和审计。P95 使用 nearest-rank，失败响应也计入耗时，非 200 数量单列；不含真实网络和模型推理。

检索基线是“标题 + 正文的不重复词项重合数量”，与服务 BM25（含标题加权）使用相同分词、语料、查询与 K。本次比较反映整套排序配置，不能单独归因于某一因素，也不是原版 LibreChat 的成绩。基线与目标相同就如实呈现，不宣称不存在的提升。

阶段 1 报告记录语料、查询、服务代码和评测器摘要、运行环境与逐查询结果。这些进程内耗时不能写成真实 Agent 问答耗时；模型 token、费用与任务结果应读取阶段 2 的独立报告。

## 阶段三来源引用校验

当前 `agent.py` 使用 `knowledge_judge.py`（3.1），原 `judge.py`（2.2）保留。新判分器修正字段名/完整分块编号被当作假文档的问题，并逐次验证来源快照、全文 SHA、精确子串、行号、版本和可信来源 URL；保留原安全、权限、审批与禁回显检查。自然语言事实支持仍需复核；自动来源检查仅能证明种子 KB 文档虚拟 v1，导入版本缺少评测快照时明确失败。

`knowledge_regrade.py --input-report <原report.json> --output <全新目录>` 只离线复算已记录证据，不调用模型、不覆盖旧报告，保留原判断和来源摘要。结果为事后判分适配，不是一次新独立评测。当前支持可信实验来源 `http://localhost:8091`，不能将工具自报 host 当作可信配置。

本轮8题旧严格规则1/8，新引用适配后6/8；关键词失败N06/M01保留。另行助手辅助语义复核也为6/8，但需修改的是K02/K09的来源归属，两个比例不应混淆。原100题95/100和人工复核记录不变。检索固定集定义、已见回归与新16题留出区别见 [../KNOWLEDGE.md](../KNOWLEDGE.md)。
# 公开资料语义评审版本说明

`public-judge-v1.3` 进一步明确：评审引文必须是证据原始 `text` 经 JSON 解码后的连续子串，不能使用 Markdown 渲染后的文字，也不能删除链接、归一化标点或拼接不连续片段。优先选择短且足以证明事实的原文；需要带链接的证据时保留完整原始格式。严格逐字校验未放宽，格式不符仍然不评分；本版不回写旧报告。

`public-judge-v1.2` 将完整度限定为最终用户问题结合对话历史所需的关键事实、步骤、适用条件和必要限定。参考答案中的可选背景、额外例子或旁支比较，不自动成为必答项；省略会改变适用性或导致操作不完整的前提仍然必须保留。判为 `partial` / `absent` 时，评审模型必须说明遗漏的必要内容及其与用户问题的关系，并检查候选是否已用改写或操作步骤表达该内容。

此次仅明确通用评审提示词并升级版本，引用、证据支持、缺证处置和本地通过条件不变。旧版本报告保留，新版本另建报告；开发样本复评不等于正式集盲测或人工验收。规则单测验证的是解析与约束，不证明评审模型必然正确执行语义规则。

