"""飞书云盘 / 消息资源（drive + im:resource）。

P2 实现：下载消息里的文件附件（精读模块收 PDF 用）。
P3 预留：云盘文件上传/下载（接口占位保留）。
"""

from __future__ import annotations

import logging
from pathlib import Path

import lark_oapi as lark
from lark_oapi.api.drive.v1 import UploadAllMediaRequest, UploadAllMediaRequestBody
from lark_oapi.api.im.v1 import GetMessageResourceRequest

logger = logging.getLogger("feishu_assistant.drive")


class DriveError(Exception):
    """云盘/资源操作失败，信息为中文（含 code/msg/log_id）。"""


class DriveClient:
    def __init__(self, client: lark.Client) -> None:
        self.client = client

    def download_message_resource(
        self,
        message_id: str,
        file_key: str,
        dest_path: Path | str,
        resource_type: str = "file",
    ) -> Path:
        """下载消息附件（im:resource）到本地，返回路径。

        message_id / file_key 均来自 im.message.receive_v1 事件的 message.content。
        """
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = (
            GetMessageResourceRequest.builder()
            .message_id(message_id)
            .file_key(file_key)
            .type(resource_type)
            .build()
        )
        resp = self.client.im.v1.message_resource.get(req)
        if not resp.success():
            raise DriveError(
                f"下载消息附件失败：code={resp.code}, msg={resp.msg}, log_id={resp.get_log_id()}。"
                "常见原因：应用未开通 im:resource 权限或未发布新版本。"
            )
        with open(dest, "wb") as f:
            if hasattr(resp.file, "write_to"):  # 旧版 SDK 的流式接口
                resp.file.write_to(f)
            else:  # 新版 SDK 直接返回 BytesIO
                f.write(resp.file.read())
        logger.info("消息附件已下载：%s（%d 字节）", dest, dest.stat().st_size)
        return dest

    def upload_media(
        self,
        file_path: Path | str,
        parent_type: str,
        parent_node: str,
    ) -> str:
        """上传媒体文件（docx 插图等），返回 file_token。

        parent_type：docx_image（文档图片块）等；parent_node：文档 document_id。
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
                    .parent_type(parent_type)
                    .parent_node(parent_node)
                    .size(size)
                    .file(f)
                    .build()
                )
                .build()
            )
            resp = self.client.drive.v1.media.upload_all(req)
        if not resp.success():
            raise DriveError(
                f"上传媒体失败：code={resp.code}, msg={resp.msg}, log_id={resp.get_log_id()}。"
                "常见原因：未开通 drive:drive 权限或未发布新版本。"
            )
        logger.info("媒体已上传：%s → %s", path.name, resp.data.file_token)
        return resp.data.file_token
