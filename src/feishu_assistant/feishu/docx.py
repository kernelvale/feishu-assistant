"""飞书云文档（docx）：精读笔记落地为飞书文档。

P2 实现：
- create_document：创建文档，返回 (document_id, url)
- write_markdown：markdown → docx block 的转换与顺序写入
  （标题/段落/无序/有序列表/加粗/行内代码/链接/表格；表格走 descendant 嵌套接口）
- grant_access：文档权限处理——组织内可读 + 给请求者 open_id 加编辑权限
  （新建文档默认只有应用可见，不处理用户点不开链接，这是最常见的坑）
- export_text / download_image：反向读取——把文档 blocks 拼回纯文本
  （图片块保留 [[IMG:token]] 位置标记），供文档评审等下游消费

markdown 转换与 blocks→文本拼接全部为纯函数（make_* / markdown_to_units /
build_table_descendants / blocks_to_text），不碰网络，可离线测试。
Block 直接用 SDK 模型的 dict-init 构造。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import lark_oapi as lark
from lark_oapi.api.docx.v1 import (
    Block,
    CreateDocumentBlockChildrenRequest,
    CreateDocumentBlockChildrenRequestBody,
    CreateDocumentBlockDescendantRequest,
    CreateDocumentBlockDescendantRequestBody,
    CreateDocumentRequest,
    CreateDocumentRequestBody,
)
from lark_oapi.api.drive.v1 import (
    BaseMember,
    CreatePermissionMemberRequest,
    PatchPermissionPublicRequest,
    PermissionPublicRequest,
)

logger = logging.getLogger("feishu_assistant.docx")

# block_type 编码：2=段落 3..8=heading1..6 12=无序列表 13=有序列表 31=表格 32=表格单元格
BLOCK_TYPE_TEXT = 2
BLOCK_TYPE_HEADING = {1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}
BLOCK_TYPE_BULLET = 12
BLOCK_TYPE_ORDERED = 13
BLOCK_TYPE_TABLE = 31
BLOCK_TYPE_TABLE_CELL = 32
_HEADING_FIELD = {1: "heading1", 2: "heading2", 3: "heading3", 4: "heading4", 5: "heading5", 6: "heading6"}

_BATCH_SIZE = 40  # docx 批量建 block 单次上限 50，留余量

# 一个写入单元：("blocks", [block dict...]) 顺序块；("table", rows) 一张表；
# ("image", label) 一张图片（label 对应调用方传入的 image_tokens）
Unit = tuple[str, Any]

# 单元格文本超过该长度时，整张表降级为段落式列表（差评根因：长文本硬塞单元格）
TABLE_CELL_PROTECT_CHARS = 150

_IMAGE_MARK_RE = re.compile(r"^\[\[(?:FIG|MATH):(.+?)\]\]$")

# Feishu code 块 language 枚举（常用子集，未命中回落 1=PlainText）
_CODE_LANG = {
    "": 1, "plaintext": 1, "text": 1, "python": 49, "py": 49, "json": 28,
    "bash": 7, "sh": 60, "shell": 60, "zsh": 60, "yaml": 67, "yml": 67,
    "xml": 66, "html": 24, "css": 12, "sql": 56, "javascript": 30, "js": 30,
    "typescript": 63, "ts": 63, "go": 22, "rust": 53, "c": 10, "cpp": 9,
    "c++": 9, "java": 29, "markdown": 39, "md": 39, "latex": 33, "r": 50,
    "dockerfile": 18, "toml": 75, "ini": 73, "properties": 73, "diff": 69,
}

BLOCK_TYPE_CODE = 14
BLOCK_TYPE_QUOTE = 15
BLOCK_TYPE_DIVIDER = 22


def make_code_block(code: str, lang: str = "") -> dict:
    """围栏代码 → 真正的代码块（等宽字体 + 语法高亮），不再是普通段落。"""
    return {
        "block_type": BLOCK_TYPE_CODE,
        "code": {
            "style": {"language": _CODE_LANG.get(lang.lower(), 1)},
            "elements": [{"text_run": {"content": code or " "}}],
        },
    }


def _is_standalone_bold(stripped: str) -> str | None:
    """整行只有一段加粗（**小节标题**）→ 返回纯文本，否则 None。"""
    m = re.fullmatch(r"\*\*([^*]{1,60})\*\*", stripped)
    return m.group(1).strip() if m else None


def image_dimensions(path: Path) -> tuple[int, int]:
    """读图片宽高；失败返回 (0, 0)。"""
    try:
        from PIL import Image  # 惰性导入

        with Image.open(path) as im:
            return im.size
    except Exception:
        return 0, 0


def fit_image_size(width: int, height: int, page_width: int = 640) -> tuple[int, int]:
    """图片块尺寸：宽约页面 90%（≤640px），等比缩放；读不到尺寸给默认。"""
    if width <= 0 or height <= 0:
        return page_width, int(page_width * 0.6)
    w = min(width, page_width)
    h = max(1, int(height * w / width))
    return w, h


class DocxError(Exception):
    """云文档操作失败，信息为中文（含 code/msg/log_id）。"""


# ---------------------------------------------------------------------------
# markdown → block 纯函数
# ---------------------------------------------------------------------------

_INLINE_RE = re.compile(r"(\*\*.+?\*\*|`[^`\n]+`|\[[^\]\n]+\]\([^)\n]+\))")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def make_text_elements(text: str) -> list[dict]:
    """把一行富文本切成 docx TextElement：支持 **加粗**、`行内代码`、[链接](url)。"""
    elements: list[dict] = []
    for part in _INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            elements.append({"text_run": {"content": part[2:-2], "text_element_style": {"bold": True}}})
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            elements.append({"text_run": {"content": part[1:-1], "text_element_style": {"inline_code": True}}})
        else:
            m = _LINK_RE.fullmatch(part)
            if m:
                label, url = m.group(1), quote(m.group(2), safe=":/?#@!$&'()*+,;=%-._~")
                elements.append({"text_run": {"content": label, "text_element_style": {"link": {"url": url}}}})
            else:
                elements.append({"text_run": {"content": part}})
    return elements or [{"text_run": {"content": ""}}]


def make_textish_block(block_type: int, content: str, indentation_level: int = 0) -> dict:
    """段落/标题/列表项都是 Text 形态，只是 block_type 与字段名不同。"""
    if block_type == BLOCK_TYPE_TEXT:
        field = "text"
    elif block_type == BLOCK_TYPE_BULLET:
        field = "bullet"
    elif block_type == BLOCK_TYPE_ORDERED:
        field = "ordered"
    else:  # 标题：block_type 3..8 → heading1..6
        field = _HEADING_FIELD[block_type - 2]
    body: dict[str, Any] = {"elements": make_text_elements(content)}
    if block_type in (BLOCK_TYPE_BULLET, BLOCK_TYPE_ORDERED) and indentation_level > 0:
        body["style"] = {"indentation_level": str(min(indentation_level, 3))}
    return {"block_type": block_type, field: body}


def _list_indentation(prefix: str) -> int:
    """Markdown 列表缩进转飞书层级；2~4 个空格均视作下一层。"""
    spaces = len(prefix.replace("\t", "    "))
    if spaces <= 0:
        return 0
    return min(3, max(1, (spaces + 1) // 2))


def split_table_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_table_separator(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{2,}:?", c or "") is not None for c in cells) and bool(cells)


def _parse_table_block(lines: list[str]) -> list[list[str]]:
    """把 markdown 表格行解析成 rows（含表头；丢弃分隔行；列数对齐）。"""
    rows = [split_table_row(ln) for ln in lines]
    rows = [r for i, r in enumerate(rows) if not (i == 1 and _is_table_separator(r))]
    width = max(len(r) for r in rows)
    return [r + [""] * (width - len(r)) for r in rows]


def _table_to_bullets(rows: list[list[str]]) -> list[dict]:
    """长单元格表格降级为段落式列表：每行一个 bullet，首格加粗引导。"""
    blocks: list[dict] = []
    header, data_rows = rows[0], rows[1:]
    for row in data_rows:
        parts = [f"**{row[0]}**"]
        for i, cell in enumerate(row[1:], 1):
            label = header[i] if i < len(header) and header[i] else f"列{i}"
            parts.append(f"{label}：{cell}")
        blocks.append(make_textish_block(BLOCK_TYPE_BULLET, "；".join(parts)))
    return blocks


def _table_needs_protection(rows: list[list[str]]) -> bool:
    return any(len(cell) > TABLE_CELL_PROTECT_CHARS for row in rows for cell in row)


def markdown_to_units(md: str) -> list[Unit]:
    """把 markdown 切成有序写入单元。支持标题/列表/表格/图片标记/代码围栏/段落。"""
    units: list[Unit] = []
    simple: list[dict] = []
    lines = md.splitlines()
    i = 0

    def flush_simple() -> None:
        nonlocal simple
        if simple:
            units.append(("blocks", simple))
            simple = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):  # 代码围栏 → 真正的代码块（含语言）
            lang = stripped[3:].strip()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # 跳过收尾围栏（未闭合时自然到 EOF）
            flush_simple()
            units.append(("blocks", [make_code_block("\n".join(code_lines), lang)]))
            continue
        image_mark = _IMAGE_MARK_RE.match(stripped)  # [[FIG:label]] / [[MATH:label]]
        if image_mark:
            flush_simple()
            units.append(("image", image_mark.group(1).strip()))
            i += 1
            continue
        if stripped.startswith("|") and stripped.endswith("|"):  # 表格块
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            rows = _parse_table_block(tbl_lines)
            if rows:
                flush_simple()
                if _table_needs_protection(rows):
                    units.append(("blocks", _table_to_bullets(rows)))
                else:
                    units.append(("table", rows))
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            simple.append(make_textish_block(BLOCK_TYPE_HEADING[level], m.group(2)))
        elif re.fullmatch(r"[-*_]{3,}", stripped):
            simple.append({"block_type": BLOCK_TYPE_DIVIDER, "divider": {}})
        elif stripped.startswith(">"):
            simple.append({
                "block_type": BLOCK_TYPE_QUOTE,
                "quote": {"elements": make_text_elements(stripped.lstrip("> ").strip())},
            })
        elif _is_standalone_bold(stripped) is not None:
            # Markdown 的整行加粗仍是段落，不是标题。若升级为 heading4，
            # 飞书会把所有加粗引导语收进文档目录，造成重复且失真的导航。
            simple.append(make_textish_block(BLOCK_TYPE_TEXT, stripped))
        elif (list_match := re.match(r"^(\s*)[-*]\s+(.*)$", line)):
            simple.append(make_textish_block(
                BLOCK_TYPE_BULLET, list_match.group(2), _list_indentation(list_match.group(1)),
            ))
        elif (list_match := re.match(r"^(\s*)\d+[.、]\s+(.*)$", line)):
            simple.append(make_textish_block(
                BLOCK_TYPE_ORDERED, list_match.group(2), _list_indentation(list_match.group(1)),
            ))
        elif stripped:
            simple.append(make_textish_block(BLOCK_TYPE_TEXT, stripped))
        i += 1

    flush_simple()
    return units


def _column_widths(col_size: int) -> list[int]:
    """列宽分配（像素）：首列窄放标签，其余均分剩余页宽。

    飞书文档表格不设 column_width 时均分列宽，长内容被挤碎（用户投诉「狗屎」）。
    """
    PAGE = 700
    if col_size <= 1:
        return [PAGE]
    first = 130 if col_size == 2 else (150 if col_size == 3 else 110)
    rest = max(80, (PAGE - first) // (col_size - 1))
    return [first] + [rest] * (col_size - 1)


def build_table_descendants(rows: list[list[str]], id_prefix: str = "tbl") -> tuple[list[str], list[dict]]:
    """把 rows 构建成 descendant 嵌套结构（table → cells → cell 内段落）。"""
    row_size, col_size = len(rows), len(rows[0])
    cell_ids = [f"{id_prefix}_c{i}" for i in range(row_size * col_size)]
    table_block = {
        "block_id": id_prefix,
        "block_type": BLOCK_TYPE_TABLE,
        "table": {
            "property": {
                "row_size": row_size,
                "column_size": col_size,
                "column_width": _column_widths(col_size),
                "header_row": True,
            }
        },
        "children": cell_ids,
    }
    descendants: list[dict] = [table_block]
    for idx, cell_id in enumerate(cell_ids):
        text_id = f"{cell_id}_t"
        r, c = divmod(idx, col_size)
        cell_block = {
            "block_id": cell_id,
            "block_type": BLOCK_TYPE_TABLE_CELL,
            "table_cell": {},
            "children": [text_id],
        }
        text_block = make_textish_block(BLOCK_TYPE_TEXT, rows[r][c])
        text_block["block_id"] = text_id
        descendants.extend([cell_block, text_block])
    return [id_prefix], descendants


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------

class DocxClient:
    def __init__(self, client: lark.Client) -> None:
        self.client = client

    @staticmethod
    def doc_url(document_id: str) -> str:
        return f"https://feishu.cn/docx/{document_id}"

    def _check(self, resp: Any, action: str) -> Any:
        if not resp.success():
            raise DocxError(
                f"{action}失败：code={resp.code}, msg={resp.msg}, log_id={resp.get_log_id()}"
            )
        return resp.data

    def create_document(self, title: str) -> tuple[str, str]:
        """创建空文档，返回 (document_id, url)。"""
        req = (
            CreateDocumentRequest.builder()
            .request_body(CreateDocumentRequestBody.builder().title(title).build())
            .build()
        )
        data = self._check(self.client.docx.v1.document.create(req), f"创建文档「{title}」")
        doc_id = data.document.document_id
        logger.info("文档已创建：%s（%s）", title, doc_id)
        return doc_id, self.doc_url(doc_id)

    def write_markdown(
        self,
        document_id: str,
        markdown: str,
        images: dict[str, Path | str] | None = None,
        start_index: int = 0,
        strict_images: bool = False,
    ) -> None:
        """把 markdown 按顺序写入文档（根 block 的 start_index 处开始插入）。

        images：[[FIG:label]] / [[MATH:label]] 标记 → 本地图片路径，按三步法上链；
        strict_images=True 时，缺路径或上传失败直接报错；论文精读使用该模式，避免
        发布声称“逐图讲解”却没有原图的文档。其他普通文档仍可降级为占位文字。
        start_index=0 为从头写（新文档）；传根块子块数即末尾追加（复读模式）。
        """
        units = markdown_to_units(markdown)
        index = start_index
        for kind, payload in units:
            if kind == "blocks":
                for chunk_start in range(0, len(payload), _BATCH_SIZE):
                    chunk = payload[chunk_start:chunk_start + _BATCH_SIZE]
                    req = (
                        CreateDocumentBlockChildrenRequest.builder()
                        .document_id(document_id)
                        .block_id(document_id)
                        .request_body(
                            CreateDocumentBlockChildrenRequestBody.builder()
                            .children([Block(b) for b in chunk])
                            .index(index)
                            .build()
                        )
                        .build()
                    )
                    self._check(
                        self.client.docx.v1.document_block_children.create(req),
                        f"写入文档块（index={index}）",
                    )
                    index += len(chunk)
            elif kind == "image":
                label = payload
                img_path = (images or {}).get(label)
                if img_path is None:
                    if strict_images:
                        raise DocxError(f"图片 {label} 缺少本地文件，已停止写入")
                    self._append_blocks(
                        document_id,
                        [make_textish_block(BLOCK_TYPE_TEXT, f"（图片 {label} 缺失，见原文）")],
                        index,
                    )
                else:
                    w, h = fit_image_size(*image_dimensions(Path(img_path)))
                    try:
                        self.add_image(document_id, img_path, w, h, index)
                    except DocxError as exc:
                        if strict_images:
                            raise DocxError(f"图片 {label} 上传失败，已停止写入：{exc}") from exc
                        logger.warning("图片上链失败（%s，降级为占位文字）：%s", label, exc)
                        self._append_blocks(
                            document_id,
                            [make_textish_block(BLOCK_TYPE_TEXT, f"（图片 {label} 上传失败，见原文）")],
                            index,
                        )
                index += 1
            else:  # table
                children_id, descendants = build_table_descendants(payload, id_prefix=f"tbl{index}")
                req = (
                    CreateDocumentBlockDescendantRequest.builder()
                    .document_id(document_id)
                    .block_id(document_id)
                    .request_body(
                        CreateDocumentBlockDescendantRequestBody.builder()
                        .children_id(children_id)
                        .descendants([Block(b) for b in descendants])
                        .index(index)
                        .build()
                    )
                    .build()
                )
                self._check(
                    self.client.docx.v1.document_block_descendant.create(req),
                    f"写入表格（index={index}）",
                )
                index += 1
        logger.info("文档 %s 写入完成：%d 个顶层块", document_id, index)

    def _append_blocks(self, document_id: str, blocks: list[dict], index: int) -> None:
        req = (
            CreateDocumentBlockChildrenRequest.builder()
            .document_id(document_id)
            .block_id(document_id)
            .request_body(
                CreateDocumentBlockChildrenRequestBody.builder()
                .children([Block(b) for b in blocks])
                .index(index)
                .build()
            )
            .build()
        )
        self._check(
            self.client.docx.v1.document_block_children.create(req),
            f"写入文档块（index={index}）",
        )

    def add_image(
        self, document_id: str, image_path: Path | str, width: int, height: int, index: int
    ) -> None:
        """插图三步法（实测可行的唯一链路）：

        1. 在 index 处创建空图片块（{"block_type": 27, "image": {}}）拿到图片块 block_id
        2. 以**图片块 block_id** 为 parent_node 上传媒体（parent_type=docx_image）
        3. patch 该块 replace_image 写入 file_token，并顺手写宽高
        """
        from lark_oapi.api.docx.v1 import (
            PatchDocumentBlockRequest,
            ReplaceImageRequest,
            UpdateBlockRequest,
        )

        # 1. 空图片块
        req = (
            CreateDocumentBlockChildrenRequest.builder()
            .document_id(document_id)
            .block_id(document_id)
            .request_body(
                CreateDocumentBlockChildrenRequestBody.builder()
                .children([Block({"block_type": 27, "image": {}})])
                .index(index)
                .build()
            )
            .build()
        )
        data = self._check(
            self.client.docx.v1.document_block_children.create(req),
            f"创建空图片块（index={index}）",
        )
        image_block_id = data.children[0].block_id

        # 2. 上传到图片块（parent_node=图片块 id，不是文档 id）
        from .drive import DriveClient

        token = DriveClient(self.client).upload_media(
            image_path, parent_type="docx_image", parent_node=image_block_id
        )

        # 3. replace_image 写入 token
        patch_req = (
            PatchDocumentBlockRequest.builder()
            .document_id(document_id)
            .block_id(image_block_id)
            .request_body(
                UpdateBlockRequest.builder()
                .replace_image(ReplaceImageRequest.builder().token(token).build())
                .build()
            )
            .build()
        )
        self._check(
            self.client.docx.v1.document_block.patch(patch_req),
            f"图片块写入 token（{image_block_id}）",
        )
        logger.info("图片块完成：%s ← %s", image_block_id, Path(image_path).name)

    def grant_access(self, doc_token: str, open_id: str | None = None) -> list[str]:
        """权限处理：组织内可读 + 给请求者加编辑权限。返回已生效的方式列表。

        新建文档默认只有应用自己可见，不做这步用户点开链接是 403。
        单个方式失败仅记日志不中断（尽量保证至少一种生效）。
        """
        granted: list[str] = []

        # 1) 链接分享设为「组织内获得链接可阅读」
        try:
            req = (
                PatchPermissionPublicRequest.builder()
                .token(doc_token)
                .type("docx")
                .request_body(
                    PermissionPublicRequest.builder()
                    .link_share_entity("tenant_readable")
                    .build()
                )
                .build()
            )
            self._check(self.client.drive.v1.permission_public.patch(req), "设置组织内可读")
            granted.append("tenant_readable")
        except DocxError as exc:
            logger.warning("设置组织内可读失败：%s", exc)

        # 2) 把请求者（消息事件里的 open_id）加为可编辑成员
        if open_id:
            try:
                req = (
                    CreatePermissionMemberRequest.builder()
                    .token(doc_token)
                    .type("docx")
                    .need_notification(False)
                    .request_body(
                        BaseMember.builder()
                        .member_type("openid")
                        .member_id(open_id)
                        .perm("edit")
                        .build()
                    )
                    .build()
                )
                self._check(self.client.drive.v1.permission_member.create(req), "添加请求者为文档成员")
                granted.append(f"member:{open_id}")
            except DocxError as exc:
                logger.warning("添加文档成员失败（open_id=%s）：%s", open_id, exc)

        if not granted:
            logger.error("文档 %s 权限授予全部失败，用户可能无法打开链接", doc_token)
        return granted

    # ---- 反向读取：文档 → 纯文本（文档评审等下游用）----

    def export_text(self, document_id: str) -> str:
        """拉取文档全部 blocks 并拼成纯文本（图片块保留 [[IMG:token]] 位置标记）。"""
        return blocks_to_text(self.list_blocks(document_id))

    def list_blocks(self, document_id: str, max_pages: int = 20) -> list[Any]:
        """翻页读取文档全部 blocks（含嵌套块，API 按文档顺序平铺返回）。"""
        from lark_oapi.api.docx.v1 import ListDocumentBlockRequest

        blocks: list[Any] = []
        page_token = ""
        for _ in range(max_pages):
            builder = (
                ListDocumentBlockRequest.builder()
                .document_id(document_id)
                .page_size(500)  # 单页上限
            )
            if page_token:
                builder.page_token(page_token)
            data = self._check(
                self.client.docx.v1.document_block.list(builder.build()), "读取文档块列表"
            )
            blocks.extend(data.items or [])
            if not data.has_more or not data.page_token:
                break
            page_token = data.page_token
        logger.info("文档 %s 读取完成：%d 个块", document_id, len(blocks))
        return blocks

    def download_image(self, file_token: str, dest_path: Path | str) -> Path:
        """按 file_token 下载文档图片到本地（精读文档里的图片块），返回路径。"""
        from lark_oapi.api.drive.v1 import DownloadMediaRequest

        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = DownloadMediaRequest.builder().file_token(file_token).build()
        resp = self.client.drive.v1.media.download(req)
        if not resp.success():
            raise DocxError(
                f"下载文档图片失败：code={resp.code}, msg={resp.msg}, log_id={resp.get_log_id()}"
            )
        file_obj = resp.file
        with open(dest, "wb") as f:
            if hasattr(file_obj, "write_to"):
                file_obj.write_to(f)
            else:  # SDK 部分版本直接给 BytesIO
                f.write(file_obj.read())
        logger.info("文档图片已下载：%s（%d 字节）", dest, dest.stat().st_size)
        return dest


# ---------------------------------------------------------------------------
# blocks → 纯文本 纯函数（反向读取）
# ---------------------------------------------------------------------------

# block_type → 承载文本的字段名（与写入侧对应；其余类型不产文本）
_READ_TEXT_FIELD = {
    BLOCK_TYPE_TEXT: "text",
    **{t: _HEADING_FIELD[t - 2] for t in BLOCK_TYPE_HEADING.values()},
    BLOCK_TYPE_BULLET: "bullet",
    BLOCK_TYPE_ORDERED: "ordered",
    14: "code",
    15: "quote",
    17: "todo",
}
BLOCK_TYPE_IMAGE = 27
IMG_MARK_PREFIX = "[[IMG:"  # 图片位置标记：[[IMG:file_token]]


def _elements_text(elements: Any) -> str:
    """TextElement 列表 → 拼接纯文本（只要 text_run 的 content）。"""
    parts: list[str] = []
    for el in elements or []:
        text_run = getattr(el, "text_run", None)
        content = getattr(text_run, "content", None)
        if content:
            parts.append(content)
    return "".join(parts).strip()


def blocks_to_text(blocks: list[Any]) -> str:
    """文档 blocks（SDK Block，按文档顺序平铺）→ 纯文本。

    标题转回 # 前缀、列表加 - / 递增序号 前缀、图片块留 [[IMG:token]] 位置标记；
    表格/单元格等容器块跳过（单元格内的文本块本身也在列表里，自然保留）。
    """
    lines: list[str] = []
    ordered_n = 0  # 连续有序列表重新编号（导出给评审/检索用，避免满篇 "1."）
    for block in blocks:
        block_type = getattr(block, "block_type", None)
        if block_type != BLOCK_TYPE_ORDERED:
            ordered_n = 0
        if block_type == BLOCK_TYPE_IMAGE:
            token = getattr(getattr(block, "image", None), "token", "") or ""
            lines.append(f"{IMG_MARK_PREFIX}{token}]]")
            continue
        field = _READ_TEXT_FIELD.get(block_type)  # type: ignore[arg-type]
        if not field:
            continue  # 页面/表格/容器块等不直接产文本
        text = _elements_text(getattr(getattr(block, field, None), "elements", None))
        if not text:
            continue
        if block_type in BLOCK_TYPE_HEADING.values():
            level = block_type - 2
            lines.append(f"{'#' * level} {text}")
        elif block_type == BLOCK_TYPE_BULLET:
            lines.append(f"- {text}")
        elif block_type == BLOCK_TYPE_ORDERED:
            ordered_n += 1
            lines.append(f"{ordered_n}. {text}")
        elif block_type == 15:
            lines.append(f"> {text}")
        elif block_type == 14:
            lines.append(f"```\n{text}\n```")
        else:
            lines.append(text)
    return "\n".join(lines)
