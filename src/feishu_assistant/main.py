"""入口：feishu-assistant CLI。

命令：
  feishu-assistant          启动 bot（WS 长连接 + BackgroundScheduler 定时调度器）
  feishu-assistant init     初始化多维表格：建 app + 8 张表，写回 config.yaml
  feishu-assistant check    检查配置与凭证，报告缺什么

所有用户可见输出均为中文；缺凭证时打印指引并退出（exit code 1），不抛 traceback。
"""

from __future__ import annotations

import argparse
import logging
import sys

from apscheduler.schedulers.background import BackgroundScheduler

from .config import (
    ConfigError,
    Settings,
    format_missing_guide,
    load_settings,
    missing_credentials,
    usable_candidates,
    write_bitable_config,
)
from .feishu import build_client
from .feishu.bitable import BitableDAO, BitableError, init_bitable
from .feishu.bot import FeishuBot
from .feishu.career_io import CareerIO
from .feishu.docx import DocxClient
from .kimi import KimiClient
from .modules.career import AUDIO_DIR, RESUME_DIR, TAILORED_RESUME_DIR, CareerService
from .modules.chat import ChatModule
from .modules.digest import (
    DEFAULT_EVENING_TIME,
    DEFAULT_PICKS_TIME,
    DEFAULT_TODAY_TIME,
    DEFAULT_WEEKLY_TIME,
    DigestService,
    register_jobs,
)
from .lifecycle import PaperCache, StoragePolicy, cleanup_sensitive_roots
from .modules.paper import PAPERS_DIR
from .modules.inbox import InboxService
from .modules.paper import PaperReadingService
from .modules.router import Router
from .modules.session import ContextBudget, SessionService
from .store import DedupStore, NotificationOutbox, PaperStore, SessionStore
from .vectors import PaperVectorIndex

logger = logging.getLogger("feishu_assistant.main")


def setup_logging() -> None:
    """INFO 级日志到 stdout：消息收发、路由命中、provider/model 命中与失败全程可见。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feishu-assistant",
        description="飞书个人助理 bot：Kimi 大模型 + 飞书长连接 + 多维表格数据层",
    )
    sub = parser.add_subparsers(dest="command", metavar="命令")

    sub.add_parser("start", help="启动 bot（默认命令，可省略）：WS 长连接收消息 + 定时调度器")
    sub.add_parser("init", help="初始化多维表格：自动建 app 与 8 张业务表，并迁移缺失字段")
    sub.add_parser("check", help="检查配置与凭证是否齐全")
    return parser


def load_or_exit() -> Settings:
    """加载配置；config.yaml 本身有问题时打印中文错误并退出。"""
    try:
        return load_settings()
    except ConfigError as exc:
        print(f"❌ 配置加载失败：{exc}", file=sys.stderr)
        sys.exit(1)


def require_credentials(settings: Settings) -> None:
    """缺凭证时打印中文指引并退出。"""
    missing = missing_credentials(settings)
    if missing:
        print(format_missing_guide(missing), file=sys.stderr)
        sys.exit(1)


def cmd_check() -> int:
    """检查配置：逐项报告 + provider 状态表，全绿返回 0，有缺失返回 1。"""
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"❌ 配置加载失败：{exc}")
        return 1

    print("🔍 配置检查结果：")
    ok = True

    checks = [
        ("FEISHU_APP_ID（飞书应用 ID）", settings.feishu_app_id),
        ("FEISHU_APP_SECRET（飞书应用密钥）", settings.feishu_app_secret),
    ]
    for label, value in checks:
        if value:
            print(f"  ✅ {label}：已配置")
        else:
            print(f"  ❌ {label}：缺失")
            ok = False

    # ---- 大模型 provider 状态表（只看 enabled/key 是否就位，不真实调 API）----
    print("\n  大模型 provider：")
    for p in settings.providers.values():
        if not p.enabled:
            print(f"  ⏸️  {p.name:<12} disabled（有 key 后在 config.yaml 改 enabled: true 启用）")
        elif p.api_key:
            print(f"  ✅ {p.name:<12} enabled，key 已就位（{p.api_key_env}）")
        else:
            print(f"  ❌ {p.name:<12} enabled 但缺 key（.env 的 {p.api_key_env}），选中后调用会直接报错")

    # ---- 当前模型选择 ----
    print("\n  当前全局模型（失败不会自动切换）：")
    labels = settings.routing_heavy[0].label() if settings.routing_heavy else "（未选择）"
    usable = usable_candidates(settings, heavy=True)
    print(f"  {labels}")
    if usable:
        print("      ✅ 当前选择可用")
    else:
        print("      ❌ 当前选择不可用（调用会直接报错，不会换模型）")
        ok = False

    for warning in settings.warnings:
        print(f"  ⚠️  {warning}")

    print(f"\n  ℹ️  每日推送时间：{settings.push_daily_time} / 晚间：{settings.push_evening_time}")

    if settings.bitable_ready():
        print(f"  ✅ 多维表格：已初始化（app_token={settings.bitable_app_token[:8]}…，8 张表就绪）")
    else:
        print("  ⚠️  多维表格：未初始化，凭证齐全后运行 feishu-assistant init")

    if ok:
        print("\n一切就绪，可以运行 feishu-assistant 启动 bot 🎉")
    else:
        print("\n请按 docs/p0-setup.md 补齐上面标 ❌ 的项（模板：cp .env.example .env）。")
    return 0 if ok else 1


def _migration_summary(created_tables: list[str], created_fields: dict[str, list[str]]) -> str:
    """把 schema 迁移结果整理成适合 CLI/聊天返回的短文案。"""
    parts: list[str] = []
    if created_tables:
        parts.append(f"新增表：{'、'.join(created_tables)}")
    if created_fields:
        field_parts = [f"{table}（{'、'.join(fields)}）" for table, fields in created_fields.items()]
        parts.append(f"新增字段：{'；'.join(field_parts)}")
    return "；".join(parts)


def _migrate_career_records(settings: Settings, client: object) -> str:
    """补旧求职记录的中心ID/版本类型；纯飞书元数据迁移，不调用模型。"""
    service = CareerService(settings, object(), client)
    links = service._repair_career_links()
    versions = service._repair_resume_version_chains(service.io.list_all("简历库"))
    parts = []
    if any(links.values()):
        parts.append(
            f"旧记录关联：看板 {links['boards']}、简历 {links['resumes']}、面经 {links['experiences']}"
        )
    if versions:
        parts.append(f"简历版本链 {versions}")
    return "；".join(parts)


def cmd_init(settings: Settings) -> int:
    """初始化多维表格并写回 config.yaml；已初始化则幂等迁移表与字段。"""
    require_credentials(settings)
    client = build_client(settings)

    if settings.bitable_ready():
        from .feishu.bitable import ensure_tables

        try:
            tables, created_tables, created_fields = ensure_tables(
                client, settings.bitable_app_token
            )
        except BitableError as exc:
            print(f"❌ 多维表格 schema 迁移失败：{exc}", file=sys.stderr)
            return 1
        merged = {**settings.bitable_tables, **tables}
        if created_tables:
            try:
                write_bitable_config(settings.bitable_app_token, merged)
            except ConfigError as exc:
                print(f"❌ 写回 config.yaml 失败：{exc}", file=sys.stderr)
                print(f"tables={merged}", file=sys.stderr)
                return 1
        settings.bitable_tables = merged
        try:
            record_summary = _migrate_career_records(settings, client)
        except BitableError as exc:
            print(f"❌ 求职旧记录迁移失败：{exc}", file=sys.stderr)
            return 1
        if not created_tables and not created_fields and not record_summary:
            print("ℹ️  多维表格及求职关联已是最新结构，无需操作。")
            return 0
        summaries = [s for s in (
            _migration_summary(created_tables, created_fields), record_summary,
        ) if s]
        print(f"✅ 多维表格已更新：{'；'.join(summaries)}。")
        return 0

    print("开始初始化多维表格：创建 app 与 8 张业务表……")
    try:
        app_token, tables = init_bitable(client)
    except BitableError as exc:
        print(f"❌ 初始化失败：{exc}", file=sys.stderr)
        print("常见原因：应用未开通 bitable:app 权限，或版本未发布。详见 docs/p0-setup.md。", file=sys.stderr)
        return 1

    try:
        write_bitable_config(app_token, tables)
    except ConfigError as exc:
        print(f"❌ 写回 config.yaml 失败：{exc}", file=sys.stderr)
        print(f"app_token={app_token}，tables={tables}", file=sys.stderr)
        return 1

    print(f"✅ 初始化完成！app_token={app_token}")
    for name, table_id in tables.items():
        print(f"  • {name}：{table_id}")
    print("已写回 config.yaml 的 bitable 段。")
    return 0


def make_chat_init_fn(settings: Settings):
    """聊天内 /init 命令：同样走多维表格初始化（幂等）。"""

    def _init() -> str:
        client = build_client(settings)
        if settings.bitable_ready():
            from .feishu.bitable import ensure_tables

            try:
                tables, created_tables, created_fields = ensure_tables(
                    client, settings.bitable_app_token
                )
            except BitableError as exc:
                return f"多维表格 schema 迁移失败：{exc}"
            merged = {**settings.bitable_tables, **tables}
            if created_tables:
                try:
                    write_bitable_config(settings.bitable_app_token, merged)
                except ConfigError as exc:
                    return f"写回 config.yaml 失败：{exc}"
            settings.bitable_tables = merged
            try:
                record_summary = _migrate_career_records(settings, client)
            except BitableError as exc:
                return f"求职旧记录迁移失败：{exc}"
            if not created_tables and not created_fields and not record_summary:
                return "多维表格及求职关联已经是最新结构，无需重复操作～"
            summaries = [s for s in (
                _migration_summary(created_tables, created_fields), record_summary,
            ) if s]
            return f"✅ 多维表格已更新：{'；'.join(summaries)}。"
        try:
            app_token, tables = init_bitable(client)
            write_bitable_config(app_token, tables)
        except (BitableError, ConfigError) as exc:
            return f"初始化失败：{exc}"
        settings.bitable_app_token = app_token
        settings.bitable_tables = tables
        return "✅ 多维表格初始化完成！已创建论文库、收藏库、灵感库、求职看板、面经库、简历库、备忘库、精选库 8 张表。"

    return _init


def cmd_start(settings: Settings) -> int:
    """启动 bot：BackgroundScheduler + WS 长连接（阻塞）。"""
    require_credentials(settings)
    for warning in settings.warnings:
        print(f"⚠️  {warning}")

    lark_client = build_client(settings)
    kimi = KimiClient(settings)
    chat = ChatModule(kimi)
    router = Router(init_fn=make_chat_init_fn(settings), panel_tables=settings.bitable_tables)
    # 事件通知外发箱：失败/卡死/恢复类通知先落库，flusher 线程 30s 一轮必达补发
    bot = FeishuBot(settings, router, outbox=NotificationOutbox())

    # 消息幂等：私聊消息去重（至少一次投递 + 重启重投）
    bot.dedup = DedupStore()
    bot.start_outbox_flusher()

    # 精读服务：router 的 paper/followup 钩子 + bot 的主动推送能力 + 向量联动
    vector_index = PaperVectorIndex()
    paper_store = PaperStore()
    storage_policy = StoragePolicy.from_env()
    paper_cache = PaperCache(PAPERS_DIR, policy=storage_policy, store=paper_store)
    reconciled_paths = paper_store.clear_missing_pdf_paths()
    startup_storage = paper_cache.cleanup(dry_run=True, include_legacy=True)
    logger.info(
        "存储启动扫描：论文对象=%d（历史目录=%d），占用=%.1fMB，失效数据库路径=%d",
        startup_storage.scanned_entries, startup_storage.legacy_entries,
        startup_storage.remaining_bytes / 1024 / 1024, reconciled_paths,
    )
    paper_service = PaperReadingService(
        settings, kimi, lark_client, paper_store,
        sender=bot, max_input_chars=settings.max_input_chars,
        vector_index=vector_index, paper_cache=paper_cache, outbox=bot.outbox,
    )
    bot.paper_service = paper_service
    router.paper_fn = paper_service.try_handle_text
    router.paper_svc = paper_service  # 「精读 N」序号引用用

    # 断线/重启恢复：上次进程死在中途的精读任务（reading_tasks 里 running 的）
    # 自动重跑并通知用户；单条失败不影响其他任务与启动（resume 内部已兜底）。
    resumed_count = paper_service.resume_interrupted()
    if resumed_count:
        logger.info("启动恢复：%d 个中断的精读任务已自动重跑", resumed_count)

    # 模型切换服务：「模型」命令查看/切换全局唯一模型（卡片面板 + 回调）
    from feishu_assistant.modules.models import ModelService
    model_service = ModelService(settings)
    model_service.sender = bot  # 「模型」发交互卡片
    bot.model_service = model_service  # 卡片下拉回调
    router.model_svc = model_service
    router.followup_fn = paper_service.answer_followup

    # 会话系统：自由对话入口改为会话服务；与精读服务互注
    session_service = SessionService(
        kimi, SessionStore(),
        budget=ContextBudget.from_dict(settings.session_budget),
        persona=chat.system_prompt,
    )
    router.session_svc = session_service
    router.chat_fn = session_service.chat  # 覆盖单轮 chat，换带记忆的会话对话
    session_service.paper_qa_fn = paper_service.answer_in_session
    paper_service.on_done = session_service.bind_paper_session

    # P3 求职模块：看板/JD 分析/简历库/面经库/录音复盘/押题/模拟面试
    career_io = (
        CareerIO(BitableDAO(lark_client, settings.bitable_app_token, settings.bitable_tables))
        if settings.bitable_ready()
        else None
    )
    career_service = CareerService(
        settings, kimi, lark_client, sender=bot, session_store=session_service.store,
        outbox=bot.outbox,
    )
    router.career_svc = career_service
    router.career_fn = career_service.try_handle_text  # 面试会话消息 + JD 粘贴识别
    bot.career_service = career_service  # 简历/音频文件分流

    # P4 自动化推送 + 万能收件箱
    digest_service = DigestService(
        settings, kimi, sender=bot, io=career_io,
        docx=DocxClient(lark_client), vector_index=vector_index,
        chat_id=settings.push_chat_id,
    )
    router.digest_svc = digest_service
    inbox_service = InboxService(settings, kimi, io=career_io)
    router.inbox_fn = inbox_service.try_handle

    # lark-oapi 的 ws.Client.start() 是阻塞式的，
    # 定时任务因此用 BackgroundScheduler（后台线程），两者互不影响。
    scheduler = BackgroundScheduler()
    # 日报/晚间回顾时间读 config.yaml push 段；今日/周报/每周精选用 digest 默认值
    register_jobs(
        scheduler, digest_service,
        arxiv_time=settings.push_daily_time,
        evening_time=settings.push_evening_time,
    )

    def storage_maintenance() -> None:
        paper_report = paper_cache.cleanup(include_legacy=True)
        sensitive_report = cleanup_sensitive_roots(
            [RESUME_DIR, TAILORED_RESUME_DIR, AUDIO_DIR],
            older_than_seconds=storage_policy.sensitive_ttl_seconds,
        )
        logger.info(
            "存储清理完成：论文释放 %.1fMB/%d项（活跃跳过%d，剩余%.1fMB）；"
            "敏感临时文件释放 %.1fMB/%d项",
            paper_report.freed_bytes / 1024 / 1024, paper_report.removed_entries,
            paper_report.active_skipped, paper_report.remaining_bytes / 1024 / 1024,
            sensitive_report.freed_bytes / 1024 / 1024, sensitive_report.removed_entries,
        )

    scheduler.add_job(
        storage_maintenance, "cron", hour=3, minute=40,
        id="storage_maintenance", replace_existing=True, coalesce=True, max_instances=1,
    )
    scheduler.start()
    print(
        "⏰ 定时调度器已启动："
        f"arXiv日报 {settings.push_daily_time} / 今日提醒 {DEFAULT_TODAY_TIME} / "
        f"晚间回顾 {settings.push_evening_time} / 组会周报 周一 {DEFAULT_WEEKLY_TIME} / "
        f"每周精选 周五 {DEFAULT_PICKS_TIME}"
        f"（推送目标 push.chat_id={settings.push_chat_id or '未配置，仅命令触发'}）"
    )

    if not settings.bitable_ready():
        print("⚠️  多维表格尚未初始化，可稍后运行 feishu-assistant init（或直接在聊天里发 /init）")

    print("🤖 bot 启动中：正在建立飞书 WebSocket 长连接……（Ctrl+C 退出）")
    try:
        bot.run()  # 阻塞
    except KeyboardInterrupt:
        print("\n👋 收到退出信号，正在关闭……")
    finally:
        scheduler.shutdown(wait=False)
    return 0


def cli(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "start"

    if command == "check":
        sys.exit(cmd_check())
    if command == "init":
        sys.exit(cmd_init(load_or_exit()))
    # 默认：start
    sys.exit(cmd_start(load_or_exit()))


if __name__ == "__main__":
    cli()
