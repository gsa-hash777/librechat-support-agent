"""HTTP acceptance checks for role-scoped, human-confirmed ticket changes."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import sqlite3
from threading import Barrier

import pytest

from test_service import csrf_from, draft, login


@pytest.fixture
def users(users):
    additions = [
        ("staff_a", "客服甲", "demo", "support", True),
        ("staff_b", "客服乙", "demo", "support", True),
        ("admin", "本组织管理员", "demo", "admin", True),
        ("other_admin", "另一组织管理员", "other", "admin", True),
        ("other_staff", "另一组织客服", "other", "support", True),
        ("inactive_staff", "停用客服", "demo", "support", False),
    ]
    return [*users, *[
        {"id": identifier, "display_name": name, "org_id": organization, "role": role, "active": active,
         "password": identifier + "-test-password", "api_key": identifier + "-test-api-key-" + "x" * 32}
        for identifier, name, organization, role, active in additions
    ]]


def ticket_view(client, headers, identifier, actor="alice"):
    response = client.get(f"/api/tickets/{identifier}", headers=headers[actor])
    assert response.status_code == 200, response.text
    return response.json()


def ticket_history(client, headers, identifier, actor="alice"):
    response = client.get(f"/api/tickets/{identifier}/history", headers=headers[actor])
    assert response.status_code == 200, response.text
    return response.json()["events"]


def create_ticket(client, users, headers, owner="alice", request_id="lifecycle-ticket-0001"):
    created = draft(client, headers, owner, request_id)
    login(client, users, owner)
    path = f"/console/drafts/{created['id']}"
    csrf = csrf_from(client.get(path))
    response = client.post(path + "/confirm", data={"csrf": csrf}, follow_redirects=False)
    assert response.status_code == 303, response.text
    result = client.get(f"/api/drafts/{created['id']}", headers=headers[owner]).json()
    return ticket_view(client, headers, result["ticket_id"], owner)


@pytest.fixture
def ticket(client, users, headers):
    return create_ticket(client, users, headers)


def update_payload(action="comment", expected_version=1, request_id="lifecycle-update-0001", **changes):
    return {"action": action, "expected_version": expected_version, "request_id": request_id,
            "note": "已核对故障信息，记录本次处理步骤。", **changes}


def propose(client, headers, ticket_id, actor="alice", **changes):
    response = client.post(f"/api/tickets/{ticket_id}/updates", headers=headers[actor], json=update_payload(**changes))
    assert response.status_code == 201, response.text
    return response.json()


def approval_form(client, users, proposal, actor="alice"):
    login(client, users, actor)
    path = f"/console/updates/{proposal['id']}"
    csrf = csrf_from(client.get(path))
    return path, csrf


def approve(client, users, proposal, actor="alice"):
    path, csrf = approval_form(client, users, proposal, actor)
    response = client.post(path + "/confirm", data={"csrf": csrf}, follow_redirects=False)
    assert response.status_code == 303, response.text
    return path, csrf


def perform(client, users, headers, ticket_id, actor, action, request_id, **changes):
    current = ticket_view(client, headers, ticket_id, actor)
    proposal = propose(client, headers, ticket_id, actor, action=action, expected_version=current["version"], request_id=request_id, **changes)
    approve(client, users, proposal, actor)
    return ticket_view(client, headers, ticket_id, actor)


def assert_unchanged(client, headers, ticket_id, version, history_count, actor="alice"):
    assert ticket_view(client, headers, ticket_id, actor)["version"] == version
    assert len(ticket_history(client, headers, ticket_id, actor)) == history_count


def test_default_roles_and_staff_directory_are_server_controlled(client, headers):
    expected = {"alice": "requester", "bob": "requester", "eve": "requester", "staff_a": "support", "staff_b": "support", "admin": "admin", "other_admin": "admin"}
    for actor, role in expected.items():
        response = client.get("/api/me", headers={**headers[actor], "X-User-Role": "admin"})
        assert response.status_code == 200
        assert response.json()["role"] == role
    staff = client.get("/api/staff", headers=headers["admin"])
    assert staff.status_code == 200
    assert {item["id"] for item in staff.json()["staff"]} == {"staff_a", "staff_b"}
    other_staff = client.get("/api/staff", headers=headers["other_admin"])
    assert {item["id"] for item in other_staff.json()["staff"]} == {"other_staff"}
    for actor in ["alice", "staff_a"]:
        assert client.get("/api/staff", headers=headers[actor]).status_code == 403
    assert client.get("/api/me", headers=headers["inactive_staff"]).status_code == 401


def test_initial_ticket_visibility_and_allowed_actions(client, headers, ticket):
    assert ticket["status"] == "open"
    assert ticket["version"] == 1
    assert ticket["assignee_id"] is None
    assert set(ticket["allowed_actions"]) == {"comment"}
    initial_history = ticket_history(client, headers, ticket["id"])
    assert len(initial_history) == 1
    assert initial_history[0]["action"] == "created"
    assert initial_history[0]["version"] == 1
    assert set(ticket_view(client, headers, ticket["id"], "admin")["allowed_actions"]) == {"assign", "start", "comment"}
    for actor in ["bob", "staff_a", "staff_b", "eve", "other_admin", "other_staff"]:
        assert client.get(f"/api/tickets/{ticket['id']}", headers=headers[actor]).status_code == 404
        assert client.get(f"/api/tickets/{ticket['id']}/history", headers=headers[actor]).status_code == 404
        visible = client.get("/api/tickets", headers=headers[actor]).json()["tickets"]
        assert ticket["id"] not in {item["id"] for item in visible}


def test_full_confirmed_lifecycle_records_every_change(client, users, headers, ticket):
    identifier = ticket["id"]
    before = len(ticket_history(client, headers, identifier))
    steps = [
        ("admin", "assign", {"assignee_id": "staff_a"}, "open"),
        ("staff_a", "start", {}, "in_progress"),
        ("alice", "comment", {"note": "补充信息：故障始于客户端升级之后。"}, "in_progress"),
        ("staff_a", "resolve", {"note": "已恢复网络配置并验证可以连接。"}, "resolved"),
        ("alice", "close", {"note": "已确认问题解决，同意关闭此工单。"}, "closed"),
    ]
    for index, (actor, action, changes, expected_status) in enumerate(steps, start=1):
        result = perform(client, users, headers, identifier, actor, action, f"full-lifecycle-{index:04}", **changes)
        assert result["status"] == expected_status
        assert result["version"] == index + 1
        assert result["assignee_id"] == "staff_a"
        if action == "assign":
            assert set(ticket_view(client, headers, identifier, "staff_a")["allowed_actions"]) == {"start", "comment"}
    events = ticket_history(client, headers, identifier)
    assert len(events) == before + len(steps)
    by_version = sorted((event for event in events if event["version"] > 1), key=lambda event: event["version"])
    assert [event["action"] for event in by_version] == [item[1] for item in steps]
    assert [event["version"] for event in by_version] == [2, 3, 4, 5, 6]
    assert [event["to_status"] for event in by_version] == [item[3] for item in steps]
    assert [event["from_status"] for event in by_version] == ["open", "open", "in_progress", "in_progress", "resolved"]
    names = {user["id"]: user["display_name"] for user in users}
    assert [event["actor_name"] for event in by_version] == [names[item[0]] for item in steps]
    for actor in ["alice", "staff_a", "admin"]:
        assert ticket_view(client, headers, identifier, actor)["allowed_actions"] == []


def test_update_proposal_is_inert_until_browser_confirmation(client, headers, ticket):
    identifier = ticket["id"]
    before = len(ticket_history(client, headers, identifier))
    proposal = propose(client, headers, identifier, "admin", action="assign", assignee_id="staff_a")
    assert proposal["status"] == "pending"
    assert proposal["confirmation_url"] == f"http://testserver/console/updates/{proposal['id']}"
    assert (datetime.fromisoformat(proposal["expires_at"]) - datetime.fromisoformat(proposal["created_at"])).total_seconds() == pytest.approx(1800)
    assert ticket_view(client, headers, identifier)["assignee_id"] is None
    assert_unchanged(client, headers, identifier, 1, before)
    for actor in ["alice", "staff_a", "other_admin"]:
        assert client.get(f"/api/updates/{proposal['id']}", headers=headers[actor]).status_code == 404
    own = client.get(f"/api/updates/{proposal['id']}", headers=headers["admin"])
    assert own.status_code == 200
    assert own.json()["id"] == proposal["id"]


@pytest.mark.parametrize("extra", [{"role": "admin"}, {"user_id": "admin"}, {"actor_id": "admin"}, {"org_id": "other"}, {"approved": True}])
def test_update_body_cannot_spoof_role_identity_or_approval(client, headers, ticket, extra):
    response = client.post(f"/api/tickets/{ticket['id']}/updates", headers=headers["alice"], json=update_payload(**extra))
    assert response.status_code == 422
    assert ticket_view(client, headers, ticket["id"])["version"] == 1


@pytest.mark.parametrize("actor", ["bob", "staff_b", "other_admin", "eve"])
def test_invisible_ticket_cannot_receive_updates(client, headers, ticket, actor):
    response = client.post(f"/api/tickets/{ticket['id']}/updates", headers=headers[actor], json=update_payload())
    assert response.status_code == 404


@pytest.mark.parametrize("action", ["assign", "start", "resolve"])
def test_requester_cannot_claim_staff_actions(client, headers, ticket, action):
    extra = {"assignee_id": "staff_a"} if action == "assign" else {}
    response = client.post(f"/api/tickets/{ticket['id']}/updates", headers=headers["alice"], json=update_payload(action=action, **extra))
    assert response.status_code == 403


def test_support_can_read_own_ticket_but_cannot_treat_it_as_assigned(client, users, headers):
    owned = create_ticket(client, users, headers, "staff_b")
    assert set(owned["allowed_actions"]) == {"comment"}
    response = client.post(f"/api/tickets/{owned['id']}/updates", headers=headers["staff_b"], json=update_payload(action="start"))
    assert response.status_code == 403


@pytest.mark.parametrize("assignee", ["alice", "inactive_staff", "other_staff", "missing_staff"])
def test_assign_target_must_be_active_support_in_same_organization(client, headers, ticket, assignee):
    response = client.post(f"/api/tickets/{ticket['id']}/updates", headers=headers["admin"], json=update_payload(action="assign", assignee_id=assignee))
    assert response.status_code == 403
    assert ticket_view(client, headers, ticket["id"])["assignee_id"] is None


@pytest.mark.parametrize("changes", [{"action": "assign"}, {"assignee_id": "staff_a"}, {"expected_version": 0}, {"action": "delete"}])
def test_update_rejects_malformed_action_contract(client, headers, ticket, changes):
    response = client.post(f"/api/tickets/{ticket['id']}/updates", headers=headers["admin"], json=update_payload(**changes))
    assert response.status_code == 422


def test_only_owner_can_close_resolved_ticket(client, users, headers, ticket):
    identifier = ticket["id"]
    perform(client, users, headers, identifier, "admin", "assign", "close-assign-0001", assignee_id="staff_a")
    perform(client, users, headers, identifier, "staff_a", "start", "close-start-0001")
    resolved = perform(client, users, headers, identifier, "staff_a", "resolve", "close-resolve-0001")
    for actor in ["staff_a", "admin"]:
        response = client.post(f"/api/tickets/{identifier}/updates", headers=headers[actor], json=update_payload(action="close", expected_version=resolved["version"], request_id=f"forbidden-close-{actor}"))
        assert response.status_code == 403
    assert ticket_view(client, headers, identifier)["status"] == "resolved"


@pytest.mark.parametrize("action,actor", [("resolve", "admin"), ("close", "alice")])
def test_illegal_transition_from_open_is_rejected(client, headers, ticket, action, actor):
    response = client.post(f"/api/tickets/{ticket['id']}/updates", headers=headers[actor], json=update_payload(action=action))
    assert response.status_code == 409
    assert ticket_view(client, headers, ticket["id"])["status"] == "open"


def test_closed_ticket_rejects_further_comments_and_assignments(client, users, headers, ticket):
    identifier = ticket["id"]
    perform(client, users, headers, identifier, "admin", "start", "closed-start-0001")
    perform(client, users, headers, identifier, "admin", "resolve", "closed-resolve-0001")
    closed = perform(client, users, headers, identifier, "alice", "close", "closed-close-0001")
    for actor, action, extra in [("alice", "comment", {}), ("admin", "assign", {"assignee_id": "staff_a"})]:
        response = client.post(f"/api/tickets/{identifier}/updates", headers=headers[actor], json=update_payload(action=action, expected_version=closed["version"], request_id=f"closed-{action}-0001", **extra))
        assert response.status_code == 409


def test_update_request_replay_conflict_and_post_apply_replay(client, users, headers, ticket):
    identifier = ticket["id"]
    proposal = propose(client, headers, identifier)
    replay = client.post(f"/api/tickets/{identifier}/updates", headers=headers["alice"], json=update_payload())
    assert replay.status_code == 200
    assert replay.json()["id"] == proposal["id"]
    conflict = client.post(f"/api/tickets/{identifier}/updates", headers=headers["alice"], json=update_payload(note="不同内容不能复用相同幂等请求编号。"))
    assert conflict.status_code == 409
    approve(client, users, proposal)
    replay = client.post(f"/api/tickets/{identifier}/updates", headers=headers["alice"], json=update_payload())
    assert replay.status_code == 200
    assert replay.json()["id"] == proposal["id"]
    assert replay.json()["status"] == "applied"
    assert ticket_view(client, headers, identifier)["version"] == 2


def test_stale_expected_version_is_rejected_when_proposing(client, users, headers, ticket):
    perform(client, users, headers, ticket["id"], "alice", "comment", "version-first-0001")
    response = client.post(f"/api/tickets/{ticket['id']}/updates", headers=headers["alice"], json=update_payload(expected_version=1, request_id="version-stale-0001"))
    assert response.status_code == 409
    assert ticket_view(client, headers, ticket["id"])["version"] == 2


def test_confirmation_rechecks_version_without_overwriting_newer_work(client, users, headers, ticket):
    identifier = ticket["id"]
    before = len(ticket_history(client, headers, identifier))
    first = propose(client, headers, identifier, request_id="version-first-0001")
    stale = propose(client, headers, identifier, request_id="version-second-0001", note="另一份仍基于旧版本的处理意见。")
    approve(client, users, first)
    path, csrf = approval_form(client, users, stale)
    response = client.post(path + "/confirm", data={"csrf": csrf}, follow_redirects=False)
    assert response.status_code == 409
    assert_unchanged(client, headers, identifier, 2, before + 1)


def test_api_key_alone_cannot_execute_update(client, headers, ticket):
    proposal = propose(client, headers, ticket["id"])
    client.cookies.clear()
    response = client.post(f"/console/updates/{proposal['id']}/confirm", headers=headers["alice"], data={"csrf": "model-generated"}, follow_redirects=False)
    assert response.status_code == 401
    assert ticket_view(client, headers, ticket["id"])["version"] == 1


@pytest.mark.parametrize("csrf", [None, "incorrect", "错误令牌"])
@pytest.mark.parametrize("action", ["confirm", "cancel"])
def test_update_browser_mutation_requires_csrf(client, users, headers, ticket, csrf, action):
    proposal = propose(client, headers, ticket["id"])
    login(client, users)
    data = {} if csrf is None else {"csrf": csrf}
    response = client.post(f"/console/updates/{proposal['id']}/{action}", data=data, follow_redirects=False)
    assert response.status_code == 403
    assert client.get(f"/api/updates/{proposal['id']}", headers=headers["alice"]).json()["status"] == "pending"
    assert ticket_view(client, headers, ticket["id"])["version"] == 1


@pytest.mark.parametrize("origin", ["https://attacker.invalid", "null"])
def test_update_confirmation_rejects_foreign_origin(client, users, headers, ticket, origin):
    proposal = propose(client, headers, ticket["id"])
    path, csrf = approval_form(client, users, proposal)
    response = client.post(path + "/confirm", data={"csrf": csrf}, headers={"Origin": origin}, follow_redirects=False)
    assert response.status_code == 403
    assert ticket_view(client, headers, ticket["id"])["version"] == 1


def test_ticket_owner_cannot_approve_another_actors_proposal(client, users, headers, ticket):
    proposal = propose(client, headers, ticket["id"], "admin", action="assign", assignee_id="staff_a")
    login(client, users, "alice")
    csrf = csrf_from(client.get("/console"))
    path = f"/console/updates/{proposal['id']}"
    assert client.get(path, follow_redirects=False).status_code == 404
    assert client.post(path + "/confirm", data={"csrf": csrf}, follow_redirects=False).status_code == 404
    assert ticket_view(client, headers, ticket["id"])["assignee_id"] is None


def test_cancelled_update_cannot_be_applied(client, users, headers, ticket):
    before = len(ticket_history(client, headers, ticket["id"]))
    proposal = propose(client, headers, ticket["id"])
    path, csrf = approval_form(client, users, proposal)
    assert client.post(path + "/cancel", data={"csrf": csrf}, follow_redirects=False).status_code == 303
    assert client.get(f"/api/updates/{proposal['id']}", headers=headers["alice"]).json()["status"] == "cancelled"
    assert client.post(path + "/confirm", data={"csrf": csrf}, follow_redirects=False).status_code == 409
    assert_unchanged(client, headers, ticket["id"], 1, before)


def test_expired_update_cannot_be_applied(client, users, headers, settings, ticket):
    before = len(ticket_history(client, headers, ticket["id"]))
    proposal = propose(client, headers, ticket["id"])
    path, csrf = approval_form(client, users, proposal)
    with sqlite3.connect(settings.db_path) as connection:
        connection.execute("UPDATE ticket_updates SET expires_at=0 WHERE id=?", (proposal["id"],))
    assert client.get(f"/api/updates/{proposal['id']}", headers=headers["alice"]).json()["status"] == "expired"
    assert client.post(path + "/confirm", data={"csrf": csrf}, follow_redirects=False).status_code == 410
    assert_unchanged(client, headers, ticket["id"], 1, before)


def test_demotion_after_proposal_prevents_execution(client, users, headers, settings):
    owned = create_ticket(client, users, headers, "admin")
    before = len(ticket_history(client, headers, owned["id"], "admin"))
    proposal = propose(client, headers, owned["id"], "admin", action="assign", assignee_id="staff_a")
    path, csrf = approval_form(client, users, proposal, "admin")
    with sqlite3.connect(settings.db_path) as connection:
        connection.execute("UPDATE users SET role='requester' WHERE id='admin'")
    assert client.get("/api/me", headers=headers["admin"]).json()["role"] == "requester"
    assert client.post(path + "/confirm", data={"csrf": csrf}, follow_redirects=False).status_code == 403
    assert_unchanged(client, headers, owned["id"], 1, before, "admin")


def test_deactivation_after_proposal_invalidates_browser_execution(client, users, headers, settings, ticket):
    proposal = propose(client, headers, ticket["id"])
    path, csrf = approval_form(client, users, proposal)
    with sqlite3.connect(settings.db_path) as connection:
        connection.execute("UPDATE users SET active=0 WHERE id='alice'")
    assert client.get("/api/me", headers=headers["alice"]).status_code == 401
    assert client.post(path + "/confirm", data={"csrf": csrf}, follow_redirects=False).status_code == 401
    assert ticket_view(client, headers, ticket["id"], "admin")["version"] == 1


def test_reassignment_revokes_previous_support_visibility(client, users, headers, ticket):
    identifier = ticket["id"]
    perform(client, users, headers, identifier, "admin", "assign", "reassign-first-0001", assignee_id="staff_a")
    assert ticket_view(client, headers, identifier, "staff_a")["assignee_id"] == "staff_a"
    perform(client, users, headers, identifier, "admin", "assign", "reassign-second-0001", assignee_id="staff_b")
    assert client.get(f"/api/tickets/{identifier}", headers=headers["staff_a"]).status_code == 404
    assert client.get(f"/api/tickets/{identifier}/history", headers=headers["staff_a"]).status_code == 404
    assert ticket_view(client, headers, identifier, "staff_b")["assignee_id"] == "staff_b"


def test_reassignment_during_work_preserves_state_and_transfers_permissions(client, users, headers, ticket):
    identifier = ticket["id"]
    perform(client, users, headers, identifier, "admin", "assign", "working-assign-0001", assignee_id="staff_a")
    perform(client, users, headers, identifier, "staff_a", "start", "working-start-0001")
    changed = perform(client, users, headers, identifier, "admin", "assign", "working-reassign-0001", assignee_id="staff_b")
    assert changed["status"] == "in_progress"
    assert changed["version"] == 4
    assert client.get(f"/api/tickets/{identifier}", headers=headers["staff_a"]).status_code == 404
    assert set(ticket_view(client, headers, identifier, "staff_b")["allowed_actions"]) == {"resolve", "comment"}


def test_assignment_confirmation_rechecks_target_account(client, users, headers, settings, ticket):
    before = len(ticket_history(client, headers, ticket["id"]))
    proposal = propose(client, headers, ticket["id"], "admin", action="assign", assignee_id="staff_a")
    path, csrf = approval_form(client, users, proposal, "admin")
    with sqlite3.connect(settings.db_path) as connection:
        connection.execute("UPDATE users SET active=0 WHERE id='staff_a'")
    assert client.post(path + "/confirm", data={"csrf": csrf}, follow_redirects=False).status_code == 403
    assert ticket_view(client, headers, ticket["id"])["assignee_id"] is None
    assert_unchanged(client, headers, ticket["id"], 1, before)


def test_confirmation_uses_persisted_proposal_not_tampered_form_fields(client, users, headers, ticket):
    proposal = propose(client, headers, ticket["id"], "admin", action="assign", assignee_id="staff_a")
    path, csrf = approval_form(client, users, proposal, "admin")
    response = client.post(path + "/confirm", data={"csrf": csrf, "action": "close", "assignee_id": "other_staff", "expected_version": "999"}, follow_redirects=False)
    assert response.status_code == 303
    current = ticket_view(client, headers, ticket["id"])
    assert current["assignee_id"] == "staff_a"
    assert current["status"] == "open"
    assert current["version"] == 2


def parallel_requests(count, function):
    ready = Barrier(count)

    def worker(index):
        ready.wait(timeout=10)
        return function(index)

    with ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(worker, range(count)))


def test_concurrent_proposal_retries_create_one_pending_update(client, headers, settings, ticket):
    before = len(ticket_history(client, headers, ticket["id"]))
    responses = parallel_requests(8, lambda _: client.post(f"/api/tickets/{ticket['id']}/updates", headers=headers["alice"], json=update_payload()))
    assert sorted(response.status_code for response in responses) == [200] * 7 + [201]
    assert len({response.json()["id"] for response in responses}) == 1
    with sqlite3.connect(settings.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM ticket_updates WHERE ticket_id=?", (ticket["id"],)).fetchone()[0] == 1
    assert_unchanged(client, headers, ticket["id"], 1, before)


def test_concurrent_confirmation_adds_one_version_and_history_event(client, users, headers, ticket):
    before = len(ticket_history(client, headers, ticket["id"]))
    proposal = propose(client, headers, ticket["id"])
    path, csrf = approval_form(client, users, proposal)
    responses = parallel_requests(8, lambda _: client.post(path + "/confirm", data={"csrf": csrf}, follow_redirects=False))
    assert [response.status_code for response in responses] == [303] * 8
    assert_unchanged(client, headers, ticket["id"], 2, before + 1)


def test_concurrent_conflicting_proposals_apply_only_one(client, users, headers, ticket):
    before = len(ticket_history(client, headers, ticket["id"]))
    proposals = [propose(client, headers, ticket["id"], request_id=f"parallel-conflict-{index:04}") for index in range(2)]
    login(client, users)
    csrf = csrf_from(client.get(f"/console/updates/{proposals[0]['id']}"))
    responses = parallel_requests(2, lambda index: client.post(f"/console/updates/{proposals[index]['id']}/confirm", data={"csrf": csrf}, follow_redirects=False))
    assert sorted(response.status_code for response in responses) == [303, 409]
    assert_unchanged(client, headers, ticket["id"], 2, before + 1)


def test_proposal_preview_escapes_note_and_does_not_execute(client, users, headers, ticket):
    note = '<script>alert("unsafe")</script> 注入内容应显示为文本。'
    proposal = propose(client, headers, ticket["id"], note=note)
    login(client, users)
    page = client.get(f"/console/updates/{proposal['id']}")
    assert page.status_code == 200
    assert "<script>" not in page.text
    assert "&lt;script&gt;" in page.text
    assert page.headers["referrer-policy"] == "same-origin"
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert ticket_view(client, headers, ticket["id"])["version"] == 1


def test_lifecycle_openapi_has_no_confirmation_or_identity_override(client):
    schema = client.get("/openapi.json").json()
    for path in ["/api/staff", "/api/tickets/{ticket_id}/history", "/api/tickets/{ticket_id}/updates", "/api/updates/{update_id}"]:
        assert path in schema["paths"]
    assert all(not path.startswith("/console") for path in schema["paths"])
    operation = schema["paths"]["/api/tickets/{ticket_id}/updates"]["post"]
    reference = operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    model = schema["components"]["schemas"][reference.rsplit("/", 1)[-1]]
    assert model["additionalProperties"] is False
    assert set(model["properties"]) == {"action", "request_id", "expected_version", "note", "assignee_id"}
    assert operation["security"] == [{"SupportBearer": []}]


def test_browser_ticket_form_previews_then_confirms_assignment(client, users, headers, ticket):
    login(client, users, "admin")
    path = f"/console/tickets/{ticket['id']}"
    csrf = csrf_from(client.get(path))
    response = client.post(path + "/updates", data={
        "csrf": csrf, "action": "assign", "request_id": "browser-assignment-0001",
        "expected_version": str(ticket["version"]), "note": "请客服甲接手并排查连接故障。", "assignee_id": "staff_a",
    }, follow_redirects=False)
    assert response.status_code == 303, response.text
    preview_path = response.headers["location"]
    assert preview_path.startswith("/console/updates/")
    assert ticket_view(client, headers, ticket["id"])["assignee_id"] is None
    confirmation_csrf = csrf_from(client.get(preview_path))
    applied = client.post(preview_path + "/confirm", data={"csrf": confirmation_csrf}, follow_redirects=False)
    assert applied.status_code == 303, applied.text
    current = ticket_view(client, headers, ticket["id"])
    assert current["assignee_id"] == "staff_a"
    assert current["version"] == 2


@pytest.mark.parametrize("version", ["1.0", "not-a-version", "-1"])
def test_browser_ticket_form_rejects_invalid_version(client, users, headers, ticket, version):
    login(client, users, "alice")
    path = f"/console/tickets/{ticket['id']}"
    csrf = csrf_from(client.get(path))
    response = client.post(path + "/updates", data={
        "csrf": csrf, "action": "comment", "request_id": "browser-invalid-version-0001",
        "expected_version": version, "note": "无效版本字段不能创建可执行变更。",
    }, follow_redirects=False)
    assert response.status_code == 422
    assert ticket_view(client, headers, ticket["id"])["version"] == 1
