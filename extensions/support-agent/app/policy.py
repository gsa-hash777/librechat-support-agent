"""Executable ticket policy and its tool-facing explanation share one rule table."""
from dataclasses import dataclass


ASSIGNEE_ROLE = "support"
ACTOR_LABELS = {
    "admin": "同组织管理员",
    "executor": "同组织管理员或该工单被指派的客服",
    "owner": "工单提交人本人，管理员和客服没有代关闭特权",
    "reader": "当前有权查看该工单的账号",
}
NOT_FOUND_GUIDANCE = "404 表示资源不存在或当前账号无权访问，不能据此断言资源不存在，也不能换身份重试。"
READ_SCOPES = {
    "requester": {"code": "own_only", "all_in_org": False, "assigned": False,
                  "description": "当前员工账号只能查看本组织内由本人提交的工单。其他员工的工单不可见；不要把客服的被分配权限套用给员工。"},
    "support": {"code": "own_or_assigned", "all_in_org": False, "assigned": True,
                "description": "当前客服账号可查看本组织内本人提交的工单或分配给本人的工单。"},
    "admin": {"code": "organization", "all_in_org": True, "assigned": False,
              "description": "当前管理员账号可查看本组织工单，不能查看其他组织工单。"},
}


@dataclass(frozen=True)
class ActionRule:
    actor: str
    states: tuple[str, ...]
    denied: str
    invalid_state: str


ACTION_RULES = {
    "assign": ActionRule("admin", ("open", "in_progress"), "只有当前组织的管理员可以分配工单。", "只有待处理或处理中的工单可以分配。"),
    "start": ActionRule("executor", ("open",), "只有被分配的支持人员或管理员可以执行此操作。", "工单当前状态不允许此操作，请刷新工单。"),
    "comment": ActionRule("reader", ("open", "in_progress", "resolved"), "当前账号无权查看该工单。", "已关闭的工单不能继续添加说明。"),
    "resolve": ActionRule("executor", ("in_progress",), "只有被分配的支持人员或管理员可以执行此操作。", "工单当前状态不允许此操作，请刷新工单。"),
    "close": ActionRule("owner", ("resolved",), "只有工单提交人可以确认关闭。", "只有已解决的工单可以关闭。"),
}


def owns_record(actor, record) -> bool:
    return record["user_id"] == actor["id"]


def can_read_ticket(actor, ticket) -> bool:
    scope = READ_SCOPES.get(actor["role"], READ_SCOPES["requester"])
    return actor["org_id"] == ticket["org_id"] and (
        scope["all_in_org"] or owns_record(actor, ticket)
        or scope["assigned"] and ticket["assignee_id"] == actor["id"]
    )


def access_scope(actor) -> dict:
    scope = READ_SCOPES.get(actor["role"], READ_SCOPES["requester"])
    summaries = {"own_only": "本组织内本人提交的工单", "own_or_assigned": "本组织内本人提交或分配给本人的工单", "organization": "本组织工单"}
    safe_response = (f"已忽略不可信内容。当前业务账号为 {actor['id']}，可见范围为{summaries[scope['code']]}。"
                     "只读查询无需人工审批。请提供具体业务问题。")
    return {"ticket_scope": scope["code"], "description": scope["description"], "read_requires_human_approval": False,
            "safe_response": safe_response}


def actor_matches(rule: ActionRule, actor, ticket) -> bool:
    if not can_read_ticket(actor, ticket):
        return False
    if rule.actor == "admin":
        return actor["role"] == "admin"
    if rule.actor == "executor":
        return actor["role"] == "admin" or actor["role"] == ASSIGNEE_ROLE and ticket["assignee_id"] == actor["id"]
    if rule.actor == "owner":
        return owns_record(actor, ticket)
    return rule.actor == "reader"


def allowed_actions(actor, ticket) -> list[str]:
    return [action for action, rule in ACTION_RULES.items()
            if actor_matches(rule, actor, ticket) and ticket["status"] in rule.states]


def action_error(actor, ticket, action: str) -> tuple[int, str] | None:
    rule = ACTION_RULES.get(action)
    if rule is None:
        return 422, "不支持的工单操作。"
    if not actor_matches(rule, actor, ticket):
        return 403, rule.denied
    if ticket["status"] not in rule.states:
        return 409, rule.invalid_state
    return None


def valid_assignee(actor, assignee) -> bool:
    return bool(assignee is not None and assignee["role"] == ASSIGNEE_ROLE
                and assignee["active"] and assignee["org_id"] == actor["org_id"])


def permission_policy() -> dict:
    return {
        "ticket_read": "同组织内仅提交人、被指派的客服或管理员可查看；普通同组织成员不自动可见。没有相关人员或共享名单授权功能。",
        "actions": {name: {"actor": ACTOR_LABELS[rule.actor], "states": list(rule.states)}
                    for name, rule in ACTION_RULES.items()},
        "assign_target": {"role": ASSIGNEE_ROLE, "active": True, "same_organization": True,
                          "guidance": "普通员工不能成为负责人；只能由管理员从实际返回的客服名单中选择。"},
        "draft_read": "草稿仅创建者本人可读取和确认，管理员无例外。",
        "proposal_read_and_confirm": "变更提案仅创建者本人可读取、确认或取消，管理员无例外；可查看工单不等于可查看他人提案。",
        "execution": "模型只能创建 pending 草稿或提案；本人在独立确认页操作后才可能执行，须再次查询真实状态。",
        "read_operations": {"requires_human_approval": False,
                            "guidance": "身份、知识、工单、历史、草稿/提案状态和可分配客服名单均为只读查询；只校验访问权限，无需在独立确认页批准。人工确认只用于正式创建或变更工单，不用于查询。"},
        "not_found": NOT_FOUND_GUIDANCE,
        "configuration": "这些是服务端业务规则，没有管理员代关闭或相关人员共享的后台配置开关。",
        "unsupported_actions": "现有工单只支持上列五种动作，不支持修改已有工单的优先级(priority)、标题或描述；管理员也不能通过这些工具修改。创建草稿时的priority不等于工单创建后可改。",
    }
