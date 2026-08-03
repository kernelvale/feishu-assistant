"""P3 求职模块的飞书 IO 辅助：bitable.py 公开泛用方法之外缺的 API。

- BitableDAO 现有 add_record / list_records（首页）；本文件补齐：
  list_all（翻页全量）、update、delete
- 简历库附件字段：上传本地文件拿到 file_token（parent_type=bitable_file），
  以及按 file_token 下载附件（押题/模拟面试取简历原文用）

全部经 CareerIO 类封装，构造只依赖 BitableDAO（测试可整体注入 fake）。
错误复用 BitableError / DriveError（中文信息 + log_id）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lark_oapi as lark
from lark_oapi.api.bitable.v1 import (
    AppTableRecord,
    DeleteAppTableRecordRequest,
    ListAppTableRecordRequest,
    UpdateAppTableRecordRequest,
)
from lark_oapi.api.drive.v1 import (
    DownloadMediaRequest,
    UploadAllMediaRequest,
    UploadAllMediaRequestBody,
)

from .bitable import BitableDAO, BitableError
from .drive import DriveError

logger = logging.getLogger("feishu_assistant.career_io")

_PAGE_SIZE = 100  # bitable 单页上限 500，100 足够稳妥


class CareerIO:
    """求职模块的多维表格/云盘读写入口。table_name 用中文表名（如「求职看板」）。"""

    def __init__(self, dao: BitableDAO) -> None:
        self.dao = dao

    @property
    def client(self) -> lark.Client:
        return self.dao.client

    def _table_id(self, table_name: str) -> str:
        table_id = self.dao.tables.get(table_name, "")
        if not table_id:
            raise BitableError(f"找不到表「{table_name}」的 table_id，请先运行 feishu-assistant init。")
        return table_id

    # ---- 记录：翻页全量读取 / 更新 / 删除（DAO 缺的三块）----

    def list_all(self, table_name: str, max_pages: int = 20) -> list[AppTableRecord]:
        """翻页读取表内全部记录（求职看板/面经库规模小，全量后内存筛选即可）。"""
        records: list[AppTableRecord] = []
        page_token = ""
        for _ in range(max_pages):
            builder = (
                ListAppTableRecordRequest.builder()
                .app_token(self.dao.app_token)
                .table_id(self._table_id(table_name))
                .page_size(_PAGE_SIZE)
            )
            if page_token:
                builder.page_token(page_token)
            resp = self.client.bitable.v1.app_table_record.list(builder.build())
            if not resp.success():
                raise BitableError(
                    f"表「{table_name}」读取记录失败：code={resp.code}, msg={resp.msg}, "
                    f"log_id={resp.get_log_id()}"
                )
            data = resp.data
            records.extend(data.items or [])
            if not data.has_more or not data.page_token:
                break
            page_token = data.page_token
        return records

    def update(self, table_name: str, record_id: str, fields: dict[str, Any]) -> None:
        """更新一条记录的部分字段。fields 的键为中文段名，值需符合字段类型。"""
        req = (
            UpdateAppTableRecordRequest.builder()
            .app_token(self.dao.app_token)
            .table_id(self._table_id(table_name))
            .record_id(record_id)
            .request_body(AppTableRecord.builder().fields(fields).build())
            .build()
        )
        resp = self.client.bitable.v1.app_table_record.update(req)
        if not resp.success():
            raise BitableError(
                f"表「{table_name}」更新记录失败：code={resp.code}, msg={resp.msg}, "
                f"log_id={resp.get_log_id()}"
            )
        logger.info("表「%s」记录已更新：%s（字段 %s）", table_name, record_id, "、".join(fields))

    def delete(self, table_name: str, record_id: str) -> None:
        """删除一条记录。"""
        req = (
            DeleteAppTableRecordRequest.builder()
            .app_token(self.dao.app_token)
            .table_id(self._table_id(table_name))
            .record_id(record_id)
            .build()
        )
        resp = self.client.bitable.v1.app_table_record.delete(req)
        if not resp.success():
            raise BitableError(
                f"表「{table_name}」删除记录失败：code={resp.code}, msg={resp.msg}, "
                f"log_id={resp.get_log_id()}"
            )
        logger.info("表「%s」记录已删除：%s", table_name, record_id)

    # ---- 简历库附件：上传 / 下载 ----

    def upload_attachment(self, file_path: Path | str) -> str:
        """上传本地文件为 bitable 附件素材，返回 file_token。

        parent_type=bitable_file、parent_node=多维表格 app_token，
        写入附件字段时用 [{"file_token": ...}]。
        """
        path = Path(file_path)
        if not path.exists():
            raise DriveError(f"上传文件不存在：{path}")
        size = path.stat().st_size
        with open(path, "rb") as f:
            req = (
                UploadAllMediaRequest.builder()
                .request_body(
                    UploadAllMediaRequestBody.builder()
                    .file_name(path.name)
                    .parent_type("bitable_file")
                    .parent_node(self.dao.app_token)
                    .size(size)
                    .file(f)
                    .build()
                )
                .build()
            )
            resp = self.client.drive.v1.media.upload_all(req)
        if not resp.success():
            raise DriveError(
                f"上传简历附件失败：code={resp.code}, msg={resp.msg}, log_id={resp.get_log_id()}。"
                "常见原因：未开通 drive:drive 权限或未发布新版本。"
            )
        logger.info("附件已上传：%s → %s", path.name, resp.data.file_token)
        return resp.data.file_token

    def download_media(self, file_token: str, dest_path: Path | str) -> Path:
        """按 file_token 下载附件素材到本地（取简历原文用），返回路径。"""
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = DownloadMediaRequest.builder().file_token(file_token).build()
        resp = self.client.drive.v1.media.download(req)
        if not resp.success():
            raise DriveError(
                f"下载附件失败：code={resp.code}, msg={resp.msg}, log_id={resp.get_log_id()}"
            )
        with open(dest, "wb") as f:
            if hasattr(resp.file, "write_to"):  # 旧版 SDK 的流式接口
                resp.file.write_to(f)
            else:  # 新版 SDK 直接返回 BytesIO
                f.write(resp.file.read())
        logger.info("附件已下载：%s（%d 字节）", dest, dest.stat().st_size)
        return dest
