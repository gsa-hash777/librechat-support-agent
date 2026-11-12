"""Provision local demo accounts and share only USE/VIEW access to the same Agent.

No model calls. Passwords are random and retained only in ignored .local files.
Service-desk roles never elevate LibreChat accounts to global administrators.
"""
import argparse
import json
from pathlib import Path
import secrets
import subprocess
import sys

from bootstrap import LocalApi, read_json, save_state, BootstrapError


def identity_id(user):
    value = user.get('id') or user.get('_id')
    if not isinstance(value, str) or not value or value == 'None':
        raise BootstrapError('LibreChat returned no valid immutable user identity.')
    return value


def login_only(api, account):
    session = api.request('POST', '/api/auth/login', {'email': account['email'], 'password': account['password']})
    if not isinstance(session, dict) or not session.get('token') or not isinstance(session.get('user'), dict):
        raise BootstrapError('Local login requires verification; finish it in LibreChat first.')
    api.token = session['token']
    return session['user']


def provision_local_accounts(root, accounts):
    if not accounts:
        return {}
    source = (Path(__file__).with_name('provision_accounts.cjs')).read_text(encoding='utf-8')
    completed = subprocess.run(['docker', 'compose', '-f', 'compose.support.yml', 'exec', '-T', 'api', 'node', '-e', source],
                               cwd=root, input=json.dumps(accounts), capture_output=True, text=True, encoding='utf-8', timeout=120)
    lines = [line.removeprefix('SUPPORT_PROVISION_RESULT=') for line in completed.stdout.splitlines() if line.startswith('SUPPORT_PROVISION_RESULT=')]
    if completed.returncode or len(lines) != 1:
        raise BootstrapError('Native local account provisioning failed; no password or role was reset.')
    return json.loads(lines[0])


def provision(root: Path, apply=False):
    owner = read_json(root / '.local/demo.json')
    users = read_json(root / '.local/users.json')['users']
    state = read_json(root / '.local/support-agent-bootstrap.json')
    account_path = root / '.local/librechat-users.json'
    config = read_json(account_path) if account_path.exists() else {'accounts': []}
    by_id = {item['support_user_id']: item for item in config['accounts']}
    active_ids = {user['id'] for user in users if user.get('active', True)}
    if owner['support_user_id'] not in active_ids:
        raise BootstrapError('The Agent owner business account must be active before provisioning.')
    for user in users:
        if not user.get('active', True):
            continue
        ident = user['id']
        if ident not in by_id:
            by_id[ident] = dict(owner) if ident == owner['support_user_id'] else {
                'support_user_id': ident, 'email': ident + '@local.test',
                'username': 'supportlab_' + ident, 'name': '服务台 · ' + user['display_name'],
                'password': secrets.token_urlsafe(24),
            }
        if len(by_id[ident]['name']) < 3 and not by_id[ident].get('librechat_user_id'):
            by_id[ident]['name'] = '服务台 · ' + user['display_name']
    if not apply:
        return {'status': 'preview', 'business_accounts': sorted(active_ids), 'agent_id': state['agent_id'],
                'permission': 'agent_viewer', 'public_sharing': False, 'model_called': False}
    config['accounts'] = list(by_id.values())
    save_state(account_path, config)  # Keep credentials even if registration stops halfway.
    created = provision_local_accounts(root, [account for ident, account in by_id.items()
                                             if ident in active_ids and ident != owner['support_user_id'] and not account.get('librechat_user_id')])
    for ident, uid in created.items():
        by_id[ident]['librechat_user_id'] = identity_id({'id': uid})
    save_state(account_path, config)
    owner_api = LocalApi('http://127.0.0.1:3080')
    owner_user = login_only(owner_api, owner)
    owner_id = identity_id(owner_user)
    agent = owner_api.request('GET', '/api/agents/' + state['agent_id'] + '/expanded')
    resource_id = agent.get('_id')
    if not isinstance(resource_id, str) or not __import__('re').fullmatch(r'[0-9a-f]{24}', resource_id):
        raise BootstrapError('Agent has no valid internal resource identity for native ACL sharing.')
    principals, report, clients = [], [], []
    seen_ids = set()
    for ident, account in by_id.items():
        if ident not in active_ids:
            continue
        if ident == owner['support_user_id']:
            api, user = owner_api, owner_user
        else:
            api = LocalApi('http://127.0.0.1:3080')
            user = login_only(api, account)
        clients.append(api)
        uid = identity_id(user)
        if uid in seen_ids:
            raise BootstrapError('Two business accounts resolve to the same LibreChat identity; no permissions changed.')
        seen_ids.add(uid)
        if uid != owner_id and user.get('role') != 'USER':
            raise BootstrapError('A demo account has an unexpected global role; no permissions changed.')
        if account.get('librechat_user_id') and account['librechat_user_id'] != uid:
            raise BootstrapError('A local account changed immutable LibreChat identity; review before sharing.')
        account['librechat_user_id'] = uid
        if uid != owner_id:
            principals.append({'type': 'user', 'id': uid, 'accessRoleId': 'agent_viewer'})
        report.append({'business_account': ident, 'librechat_role': user.get('role'),
                       'agent_access': 'owner' if uid == owner_id else 'viewer'})
    save_state(account_path, config)
    removed = [{'type': 'user', 'id': account['librechat_user_id']} for ident, account in by_id.items()
               if ident not in active_ids and account.get('librechat_user_id') and account['librechat_user_id'] != owner_id]
    if principals or removed:
        owner_api.request('PUT', '/api/permissions/agent/' + resource_id, {
            'updated': principals, 'removed': removed, 'public': False,
        })
    for api in clients:
        # Native ACL must allow using the same Agent for each account.
        api.request('GET', '/api/agents/' + state['agent_id'])
    return {'status': 'ready', 'accounts': report, 'agent_id': state['agent_id'],
            'public_sharing': False, 'connection': 'Each account must explicitly authorize its own business identity.',
            'model_called': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(provision(args.root, args.apply), ensure_ascii=True, indent=2))
    except (BootstrapError, ValueError, KeyError, OSError) as error:
        diagnostic = str(error) if isinstance(error, BootstrapError) else type(error).__name__
        print('Multiuser setup failed; existing passwords were retained. ' + diagnostic, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
