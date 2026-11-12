/* Local administrator provisioning. Mirrors config/create-user.js's registerUser
 * path; accepts secrets via stdin, never argv, and emits only IDs/status. */
const path = require('path');
require('module-alias')({ base: path.resolve('/app/api') });
const mongoose = require('mongoose');
const { User } = require('@librechat/data-schemas').createModels(mongoose);
const connect = require('/app/config/connect');
const { registerUser } = require('~/server/services/AuthService');

(async () => {
  let input = '';
  for await (const chunk of process.stdin) {
    input += chunk;
    if (input.length > 100000) throw new Error('invalid_input');
  }
  const accounts = JSON.parse(input);
  if (!Array.isArray(accounts) || accounts.length > 20) throw new Error('invalid_input');
  await connect();
  if (await User.countDocuments({}) === 0) throw new Error('owner_required');
  const result = {};
  for (const account of accounts) {
    if (!/^[a-zA-Z0-9_-]{1,64}$/.test(account.support_user_id) || !account.email?.endsWith('@local.test')) throw new Error('local_account_required');
    let existing = await User.findOne({ $or: [{ email: account.email }, { username: account.username }] }).lean();
    if (!existing) {
      const response = await registerUser({ email: account.email, username: account.username, name: account.name,
        password: account.password, confirm_password: account.password }, { emailVerified: true, role: 'USER', provider: 'local' });
      if (response.status !== 200 || response.userCreated !== true) throw new Error('registration_failed');
      existing = await User.findOne({ email: account.email }).lean();
    }
    if (!existing || existing.email !== account.email || existing.username !== account.username || existing.role !== 'USER') throw new Error('existing_identity_conflict');
    result[account.support_user_id] = existing._id.toString();
  }
  process.stdout.write('\nSUPPORT_PROVISION_RESULT=' + JSON.stringify(result) + '\n');
  await mongoose.disconnect();
  process.exit(0);
})().catch(() => { process.stderr.write('Local account provisioning failed.\n'); process.exit(1); });
