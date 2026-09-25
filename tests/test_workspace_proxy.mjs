import test from 'node:test';
import assert from 'node:assert/strict';

const base = process.env.WORKSPACE_URL || 'http://127.0.0.1:3000';

test('proxy rejects browser writes from a different origin before reaching backend', async () => {
  const response = await fetch(`${base}/api/backend/api/workspace/project`, {
    method: 'POST', headers: { origin: 'https://not-this-app.example', 'content-type': 'application/json' },
    body: JSON.stringify({ scene: 'not-created', name: 'Not created' }),
  });
  assert.equal(response.status, 403);
});

test('proxy accepts real browser origin through Next loopback normalization', async () => {
  const response = await fetch(`${base}/api/backend/api/workspace/project`, {
    method: 'POST', headers: { origin: base, 'content-type': 'application/json' },
    body: JSON.stringify({ scene: '../invalid', name: 'Invalid' }),
  });
  assert.equal(response.status, 400);
});

test('proxy does not forward arbitrary paths outside its API boundary', async () => {
  const response = await fetch(`${base}/api/backend/_key.pem`);
  assert.equal(response.status, 404);
});

test('runtime bridge refuses repository credentials', async () => {
  const response = await fetch(`${base}/runtime/_key.pem`);
  assert.equal(response.status, 404);
});
