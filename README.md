# feishu-assistant

跑在自己 Mac 上的**飞书个人助理 bot**：飞书 WebSocket 长连接收发消息（无需公网 IP），多维表格做数据层，多 provider 大模型手动选择（失败不自动切换）。

四大块能力：

- **论文精读**：发 arXiv 链接或 PDF，产出教学型精读飞书文档（产出优先管线，见下）
- **求职面试**：求职看板、JD 拆解、简历库/定制、面经库、录音复盘、押题、文字模拟面试
- **自动化推送**：arXiv 日报（8:30）、今日提醒（9:00）、晚间回顾（22:00）、每周精选（周五 20:00）、组会周报（周一 9:00）
- **会话系统 + 万能收件箱**：多会话隔离与跨会话召回；链接/备忘/收藏/灵感自动归档

## 论文精读管线（V9 · 叙事优先）

触发：直接发 arXiv 链接（`abs`/`pdf` 均可），或把论文 PDF 作为文件发给机器人。全程进度推送。

链路：**PDF 解析（MinerU 主，PyMuPDF4LLM 兜底）→ 全文来源导航（长文分块证据卡）→ 同一 Author 生成轻量 Narrative Brief → 同一 Author 连续生成完整初稿（长文最多 3 个宏观续写段，始终带真实前文）→ 程序机械护栏 → 一次事实与证据审查 + 一次真实读者审查 → 最多一次精确文字补丁 → 只验证实际改动 → 发布**。

设计原则：

- **作者初稿是质量基线**：`author-draft.md` 永不覆盖；没有精确、可验证的问题就不修改，修订失败或验证不可用时逐字回滚初稿
- **两审一验**：事实审查只看来源与成稿，读者审查只看文章；Reviewer 不写正文、不接触 Plan/Debt 黑话；任一审查不可用都不启动自动修订
- **限制重复的生产源**：不再分“读者章节/技术层/逐图总览”三轨写作；核心概念完整解释一次，图表一个主讲解位置，后文只允许承担新功能的短回指
- **有界修订**：Writer 只返回唯一 `old_text → new_text` 补丁；事实卡最多 3 张、读者卡最多 2 张、总计最多 5 张，整批同时受 20% 与 6000 字符双上限约束，禁止整章重写兜底
- **诚实发布**：审核不可用时记录 `fact_review_unavailable` / `reader_review_unavailable`，绝不合成 `pass` 或质量分数
- **来源覆盖**：长文原文包显式记录关键词回退、遗漏块与块内截断；依据不完整时不修订，只能发布作者草稿，不能标 `verified_final`
- **缓存**：工作区签名之外，Brief、Author、两次审查、修订和验证均绑定上游内容 hash；正文或编辑卡变化时自动失效下游缓存
- **幂等与自愈**：同一篇同时只跑一条管线；瞬时网络故障 2/5/10 分钟退避续跑（检查点不重来）；进程崩溃的任务重启后自动恢复；复读版本号 +1 并串历史版本

V8（`reading-v4`）实现仍保留作取证和显式回归；生产默认使用 V9 工作区 `reading-v5`。完整协议见 `docs/reading-pipeline-v9.md`。

产出：飞书文档 + 消息卡片 + 论文库归档（方向自动分类）+ 向量入库（bge-large-en-v1.5）+ 自动建「《标题》精读」绑定会话可追问。

## 环境要求

- macOS（Apple Silicon），无需 Docker
- [uv](https://docs.astral.sh/uv/)（自动管理 Python 3.12 与依赖，不要用系统 Python 3.9）
- 一个飞书自建应用 + 至少一个大模型 provider 的 API Key

## 从零搭建（约 20~30 分钟）

### 1. 创建飞书企业自建应用

1. 打开 [open.feishu.cn](https://open.feishu.cn)，进入**开发者后台**，**创建企业自建应用**（没有企业就免费注册一个"个人团队"，无需资质）。
2. 应用详情页左侧**添加应用能力** → 添加**机器人**。

### 2. 开通权限

左侧**权限管理**，搜索并开通（认准权限代码，中文名可能不同）：

| 权限 | 用途 |
|---|---|
| `im:message` | 接收/读取消息 |
| `im:message:send_as_bot` | 以机器人身份发消息 |
| `im:resource` | 读取消息中的图片/文件（语音、PDF 附件） |
| `bitable:app` | 多维表格读写（数据层） |
| `docx:document` | 云文档读写（精读笔记/报告） |
| `drive:drive` | 云盘文件上传下载 |

### 3. 配置事件订阅（长连接）

左侧**事件订阅** → 添加事件「**接收消息**」（`im.message.receive_v1`），订阅方式选「**使用长连接接收事件**」（WebSocket，无需公网 IP/回调地址）。

### 4. 创建版本并发布

左侧**版本管理与发布** → 创建版本 → 提交发布。⚠️ **每次改权限或事件订阅后都必须重新发版**，这是最常见的坑。

### 5. 安装与配置

```bash
git clone <本仓库> && cd feishu-assistant
uv sync                       # 自动创建 .venv（Python 3.12）并安装依赖
cp .env.example .env          # 填入密钥（见下）
cp config.example.yaml config.yaml   # 业务配置，已带中文注释
```

`.env` 需要：

```ini
FEISHU_APP_ID=cli_xxxxxxxxxxxx        # 应用详情页「凭证与基础信息」
FEISHU_APP_SECRET=xxxxxxxxxxxxxxxx    # 同页
FREE_TOKENS_API_KEY=xxxxxxxxxxxxxxxx  # 或改用其他 provider 的 key（见下节）
```

### 6. 初始化与启动

```bash
uv run feishu-assistant check   # 验证凭证与路由
uv run feishu-assistant init    # 自动建好多维表格 app + 8 张业务表，并写回 config.yaml
uv run feishu-assistant         # 启动 bot
```

然后在飞书里找到机器人发消息即可，发 `/help` 查看命令。

## 模型供给（多 provider，失败不自动切换）

支持 OpenAI 兼容协议与 Anthropic Messages 协议；可配置多个 provider，但每次只调用当前显式选中的一个模型。429/5xx/超时只在同模型内重试，耗尽后通知，绝不自动换 API 或模型。精读启动时冻结当时的选择，中途切换只影响新任务。

| provider | 协议 | .env 变量 | 说明 |
|---|---|---|---|
| free-tokens | OpenAI | `FREE_TOKENS_API_KEY` | 免费多模型网关，默认主力（kimi-k3 等） |
| kimi-coding | Anthropic | `KIMI_CODING_API_KEY` | Kimi Code 订阅端点 |
| moonshot | OpenAI | `MOONSHOT_API_KEY` | 官方按量付费 |

在 bot 里发「模型」打开面板：切换选择、在线添加 provider（key 只写入按名称派生的 .env 变量）。

## 命令表（节选）

| 口令 | 作用 |
|---|---|
| 帮助 / /help | 显示帮助 |
| 新会话 [标题] / 会话列表 / 切换 N / 改名 / 归档 N / 删除 N | 多会话管理（会话间严格隔离） |
| 面板 | 论文库多维表格链接 |
| arXiv日报 / 今日 / 周报 | 立即触发对应推送 |
| 看板 [操作] | 求职看板自然语言增删改查 |
| 定制简历 求职ID / 确认定制简历 求职ID | JD 定向简历（有证据的最小替换） |
| 押题 公司 / 模拟面试 公司 / 面经 [公司] [轮次] | 面试备战 |
| 记一下/提醒我/收藏/灵感 + 内容 | 万能收件箱归档 |

发语音或录音文件 → 自动转写（mlx-whisper，首次需 `uv add mlx-whisper`）→ 面试复盘报告。

## 测试

```bash
uv run pytest        # 全部离线运行（openai / 飞书均 mock），无凭证也应全绿
```

## 开机自启（launchd，可选）

```bash
mkdir -p logs
cp scripts/com.user.feishu-assistant.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.user.feishu-assistant.plist
```

建议先手动 `uv run feishu-assistant` 跑通确认后再装自启。

## 项目结构

```
feishu-assistant/
├── pyproject.toml          # uv 管理，CLI 入口 feishu-assistant
├── .env.example            # 密钥模板
├── config.example.yaml     # 业务配置模板（providers/路由/推送时间/bitable 映射）
├── prompts/                # 全部提示词（系统人设、精读各阶段、求职、推送）
├── scripts/                # 冒烟/运维脚本 + launchd 模板；attic/ 为一次性历史脚本
├── src/feishu_assistant/
│   ├── main.py             # 入口：start / init / check；服务实例化与 cron 注册
│   ├── config.py           # .env + config.yaml 加载与校验
│   ├── kimi.py             # 大模型客户端：显式选模、同模型重试、流式、缓存统计
│   ├── pdf.py / asr.py     # PDF 解析（MinerU）/ 语音转写（mlx-whisper 惰性）
│   ├── store.py            # sqlite：论文全文、消息去重、任务登记、会话 FTS5、通知外发箱
│   ├── feishu/             # bot.py（WS 长连接分发）、bitable.py、docx.py、drive.py、career_io.py
│   └── modules/
│       ├── router.py       # 命令注册表 + 功能钩子路由
│       ├── paper.py        # 精读任务编排：幂等、重试、恢复、发布、追问
│       ├── reading_v9.py         # 生产精读内核：单一作者、两审、一次补丁、定点验证
│       ├── reading_v9_models.py  # V9 严格数据协议（无 pass/score）
│       ├── reading_v9_quality.py # 重复候选、机械护栏与安全补丁
│       ├── reading_evidence.py   # 全文证据索引与结构化来源覆盖状态
│       ├── reading_workflow.py   # reading-v4/V8 与 reading-v5/V9 隔离工作区
│       ├── reading.py / reading_plan.py / reading_repair.py  # V8 历史兼容与取证
│       ├── session.py      # 会话系统：上下文装配、增量压缩、FTS 召回
│       ├── career.py       # 求职面试全家桶
│       ├── digest.py       # 定时推送：日报/提醒/精选/周报
│       └── inbox.py        # 万能收件箱
└── tests/                  # pytest：全离线（openai / 飞书均 mock）
```

## 常见问题

- **启动提示缺密钥**：`uv run feishu-assistant check` 会列出缺哪项，对照 `.env.example` 补齐。
- **init 报权限错误 / 改了权限不生效**：权限改动后必须重新创建版本并发布。
- **大模型 401**：当前 provider 的 key 填错/失效/没额度；系统不会自动换模型，在「模型」面板手动切换。
- **bot 收到消息没回复**：确认事件订阅为长连接模式且版本已发布；拿日志里的 `log_id` 去开放平台排查。
- **换电脑部署**：代码 + `.env` + `config.yaml` 拷过去即可，不要重复 init（会在飞书里再建一份表）。
- **数据安全**：论文 PDF 按内容哈希存 `data/`，LRU 自动清理；简历、录音等敏感材料只放任务级临时目录，结束即删。
