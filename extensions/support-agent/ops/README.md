# 多人部署、检查与备份

在 VS Code 打开根目录 `SupportAgent.code-workspace`，通过“终端 → 运行任务”执行：

| 任务 | 行为 |
| --- | --- |
| `OperationsStart` / `Support: Start` | 初始化已有配置中缺少的文件、准备本地 TLS、启用每用户 OAuth、启动业务服务并迁移 Agent Action |
| `StartDebug` | 先完成一次正常 OAuth/TLS 启动，再只为 support-agent 启用 reload 与 loopback 5678 调试端口 |
| `OpenApps` | 在默认系统浏览器打开 LibreChat 和 HTTPS 服务台；从登录到 OAuth 授权、回调均使用同一浏览器配置文件 |
| `Support: Start with hybrid retrieval options` | 要求已准备检索模型，保留 OAuth/TLS 部署配置启动 |
| `Backup` | 在线备份，不停服务；输出备份路径和摘要 |
| `Monitor` | 检查服务健康、就绪状态、工具指标、备份年龄及 Docker 资源；返回非零代表警告或错误 |
| `MultiUser` | 准备独立 LibreChat 本地账号、授予同一 Agent 的私有查看/使用权限，并验证各账号可访问；每位用户仍需自行连接服务台身份 |
| `Support: Stop` | 停止 api、support-agent、support-tls、mongodb，保留独立公共资料服务和持久化数据 |

这些启动、检查、备份和多人核验任务不调用收费模型。原来的真实 Agent 评测任务继续使用原预算与“paid tasks”标识，不会随启动自动运行。

`Start` 默认叠加 `compose.operations.yml`；发现 `.local/models/support-retrieval/manifest.json` 时，同时叠加 `compose.retrieval.yml`，没有模型则使用 lexical 配置。已有模型不会因普通启动而被换回基础镜像，检索启动也不会退回固定工具凭证。正常入口为 LibreChat `http://localhost:3080`，服务台 `https://localhost:8443`。

默认启动不开放 debugpy，也不启用自动重载。需要断点调试时执行 `StartDebug`，它在 operations 配置后叠加 `compose.debug.yml`，随后使用工作区已有的 `Support Agent: attach to running Docker service` 调试配置连接 `127.0.0.1:5678`。OAuth/TLS 设置保留，公共资料服务不变；再次执行正常 `Start` 即回到无调试/重载的部署方式。

首次启动时 `local_tls.ps1` 为当前本机生成 TLS 证书；Windows 可能显示信任提示，作用范围是当前用户的本地服务证书。后续启动会先检查证书有效期及 CurrentUser 信任存储，已信任时不再请求信任。脚本不自动覆盖半缺失或过期证书对，遇到这种情况会明确退出，需要主动轮换证书/私钥并更新信任。原本的 HTTP 8090 是本机健康检查入口，登录和 OAuth 使用 HTTPS 8443。

本轮环境验证中，Windows CurrentUser 信任存储、PowerShell HTTPS 请求和 Node 严格 TLS 检查已通过，内置 IAB 仍出现 `ERR_CERT_AUTHORITY_INVALID`。因此本机交互建议使用 `OpenApps` 在系统默认浏览器打开整套流程：先登录 LibreChat，再从该会话发起账号连接，完成服务台授权并回到 LibreChat。OAuth 回调依赖 LibreChat 登录 Cookie，不能只把授权链接从 IAB 复制到另一个浏览器而期待原会话自动完成。上述检查不是 Edge 截图或系统浏览器整套人工流程的验收记录；`OpenApps` 也仅请求打开页面，不代替授权结果验证。

`Bootstrap` 要求 `/ready` 成功、OpenAPI 工具地址为 TLS 服务、运行配置禁用旧共享 API Key。更新 OAuth 回调地址时仅重启 support-agent，随后检查就绪；不重启独立公共资料服务。多人核验任务也会先检查此部署前提。

```powershell
.\.venv\Scripts\python.exe extensions/support-agent/scripts/project.py OperationsStart
.\.venv\Scripts\python.exe extensions/support-agent/scripts/project.py OpenApps
.\.venv\Scripts\python.exe extensions/support-agent/scripts/project.py Monitor
.\.venv\Scripts\python.exe extensions/support-agent/scripts/project.py MultiUser
```

## 本地备份与恢复演练

在 `E:\LibreChat` 执行以下命令。所有备份和恢复目录默认位于已被 Git 忽略的 `.local` 内。脚本只恢复到新建目录和临时 Mongo 容器，没有覆盖线上数据库的命令。

```powershell
# 在线备份：不停止服务
.\.venv\Scripts\python.exe extensions/support-agent/scripts/backup.py create

# 停写备份：短暂停止 api、support-agent，采集完成后恢复原来运行的容器
.\.venv\Scripts\python.exe extensions/support-agent/scripts/backup.py create --quiesce

# 将上面输出的实际路径、摘要代入；摘要最好单独保存在备份目录之外
.\.venv\Scripts\python.exe extensions/support-agent/scripts/backup.py verify "<backup目录>" --manifest-sha256 <摘要>
.\.venv\Scripts\python.exe extensions/support-agent/scripts/backup.py drill "<backup目录>" --manifest-sha256 <摘要>

# 仅列出超过保留数量的完整备份；此命令不会删除任何文件
.\.venv\Scripts\python.exe extensions/support-agent/scripts/backup.py retention-plan --keep 7
```

## 备份范围与一致性

| 数据 | 实现 | 恢复检查 |
| --- | --- | --- |
| 工单、角色、审核记录、知识库导入全文、修订、分块 | `sqlite3.Connection.backup` 生成 `support.db` | SHA-256、SQLite integrity/FK 检查、表行数一致 |
| LibreChat 用户、会话、消息、Agent、Action 配置 | 现有 Mongo 镜像中的 `mongodump --db LibreChat --archive --gzip` | 无网络、无宿主端口的临时 Mongo 内执行 `mongorestore --stopOnError`，报告集合行数 |
| LibreChat 上传和运行数据 | 从 API 容器的 `/app/uploads`、`/app/data` 只读卷采集 | SHA-256、拒绝路径越界/链接/特殊文件，再恢复到新目录 |
| 项目原始知识资料 | `extensions/support-agent/data` 完整复制 | 每文件长度与 SHA-256 |

导入的 Markdown/TXT 内容存于 SQLite 的 `knowledge_revisions`，分块存于 `knowledge_chunks`，没有遗漏的独立上传文档目录。日志、可重建向量缓存、模型文件和独立 `support-public` 基准服务不在本备份范围；备份不停止或改写公共评测服务。

在线模式仅保证 SQLite 是事务一致快照。Mongo 集合和文件可能在采集过程中变化，整个包不保证同一时刻的一致性。需要演示跨组件恢复时优先使用 `--quiesce`：只暂停当前项目的 `api` 和 `support-agent`，Mongo 保持运行以执行逻辑备份。它不是分布式事务，也无法阻止外部写入或 Mongo TTL 后台删除；本地演示应避免同时运行写数据的脚本。正常退出或异常时会尝试恢复本命令停止的两个容器，之后仍应检查服务健康。进程被强制杀死、主机断电时无法保证执行清理，使用项目正常启动命令恢复。

SQLite 在 WAL 模式下即使只读查询，也可能需要建立 WAL/SHM 锁文件。因此在线备份在现有容器内用 `mode=ro` 连接执行备份 API；停写模式从只读卷将静止的 db 和剩余 WAL 复制到临时空间，再通过备份 API 输出数据库，不把未 checkpoint 的 WAL 漏掉。

最终输出的数据库会切换为 `journal_mode=DELETE`，形成无需旁侧 WAL 的封闭快照。校验与隔离恢复只对已经通过文件摘要检查的备份使用 `immutable=1`，不会读取或新建 WAL/SHM；在线源库从不使用 immutable。兼容早期备份中留下的空 WAL/SHM：若原清单包含它们仍核对原摘要，否则只允许忽略空 WAL 及对应 SHM；非空 WAL 一律拒绝。旧备份的文件和清单不因此改写。

## 私密配置需要单独保管

数据库本身含业务资料、密码/令牌哈希及加密的工具凭证。备份工具**不加密备份**，使用宿主机现有文件访问权限；Git ignore 也不是访问控制。请把备份存于仅本人可访问的位置，离机保存时使用已有的加密存储。不要提交到 Git 或作为公开简历附件。

以下文件仅在 manifest 记录位置和用途，不复制内容：

- `.env`：尤其是原 `CREDS_KEY`、`CREDS_IV`，丢失后无法解密 Mongo 中的 Action 凭证；还包括模型和会话签名密钥。
- `.local/users.json`：用户身份、角色和初始凭证配置。恢复时不要使用不同角色的旧配置覆盖数据库状态。
- `.local/operations.json`：OAuth 客户端私密配置与监控令牌。
- `.local/demo.json`：如仍使用本地初始化流程，需要保留相应登录资料。
- `.local/librechat-users.json`：各 LibreChat 账号的本地登录凭证和原生身份映射。
- `.local/support-tls/server.key` 和 `server.crt`：私钥及匹配证书应安全单独保管；也可重新生成并重新配置客户端信任。

镜像 ID、配置模板文件的 SHA-256 记录在 manifest。完整灾难恢复还需要相同代码/兼容镜像、单独保管的私密配置及 Docker 环境；当前包不是独立可启动的全部环境。恢复演练验证数据库、文件和文档可还原，没有自动替换当前部署、恢复登录会话或开展模型调用。

## 验收与边界

成功备份含 `manifest.json`，失败或中断的目录带 `.incomplete`。成功 `drill` 的 `restore-report.json` 位于新建 `.local/restore-drills/drill-*` 目录；只有完整 Mongo、SQLite、文件步骤成功才标记 `passed`。`--skip-mongo` 明确输出 `partial_pass`，不能当作完整恢复成功。

校验拒绝缺文件、多余文件、内容变化、越界路径、Windows 特殊文件名、大小写冲突、符号链接和压缩包链接。默认最多恢复 100,000 个归档成员、20 GiB 解压数据。manifest 内的摘要只检测相对该清单的变化，不能证明清单本身可信；传入独立保留的 `--manifest-sha256` 才能同时发现清单被改写。

保留策略命令只检查指定目录中的完整有效备份并输出删除候选，永不自动清理。失败目录、恢复演练目录也不会自动删除；决定清理前先确认最新备份可恢复，再核对准确目录。

```powershell
.\.venv\Scripts\python.exe -m pytest extensions/support-agent/tests/test_backup.py -q
```

自动化测试只使用临时目录和假 Docker 接口，验证 WAL 快照、越界与篡改拒绝、停写异常后恢复、原目录不被覆盖和临时容器隔离。真实 Docker 备份与恢复演练需以上述命令及实际生成的报告为准。
