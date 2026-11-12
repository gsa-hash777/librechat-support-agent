"""Object types are explicit in the Action schema; console URLs stay unchanged."""
from typing import Annotated

from fastapi import Path


TicketId = Annotated[str, Path(pattern=r"^T-[A-Za-z0-9_-]{1,80}$", description="正式工单编号 T-…；新编号为 T- 加16位大写十六进制。不得传 D- 草稿或 U- 提案编号；不要自行更改前缀。")]
DraftId = Annotated[str, Path(pattern=r"^D-[A-Za-z0-9_-]{1,80}$", description="草稿编号 D-…；新编号为 D- 加16位大写十六进制。查询是否已提交必须用 getMyTicketDraft；不是 T- 工单或 U- 提案。")]
UpdateId = Annotated[str, Path(pattern=r"^U-[A-Za-z0-9_-]{1,80}$", description="变更提案编号 U-…；新编号为 U- 加16位大写十六进制。仅提案创建者可查；不是 T- 工单或 D- 草稿。")]

OBJECT_TYPES = {
    "T-": {"kind": "正式工单", "read_tool": "getMyTicket", "history_tool": "getTicketHistory"},
    "D-": {"kind": "工单草稿", "read_tool": "getMyTicketDraft", "guidance": "草稿提交状态和最终 ticket_id 从本工具查询；不要改成 T- 前缀猜测。"},
    "U-": {"kind": "变更提案", "read_tool": "getTicketUpdate", "guidance": "pending 不代表工单已变更。"},
}
