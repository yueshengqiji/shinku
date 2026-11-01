/** 空转发回退的合成契约回归；不连接QQ、NapCat、模型或发送器。 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { recoverEmptyForward, rewriteNapCatBundle } from '../integrations/napcat-forward-recovery/index.mjs';

const EMPTY = Object.freeze({ messages: [] });
const FULL = Object.freeze({ messages: [{ sender: { user_id: 'unverified-author' }, message: [
  { type: 'text', data: { text: 'source-body-fixture' } },
] }] });
const RESOURCE = 'fixture_resource_A+/==';
const root = resource => ({ msgId: '7000000000000000101', elements: [{ multiForwardMsgElement: { resId: resource } }] });

test('原空结果只用同一RawMessage的资源ID补读一次，原作者和正文保持', async () => {
  const calls = [];
  const result = await recoverEmptyForward(EMPTY, root(RESOURCE), id => { calls.push(id); return FULL; });
  assert.equal(result, FULL); assert.deepEqual(calls, [RESOURCE]);
  assert.equal(result.messages[0].sender.user_id, 'unverified-author');
});

for (const value of [FULL, null, {}, { messages: {} }, { messages: 'unknown' }])
  test(`正常结果或非空数组形状不增加任何读取：${JSON.stringify(value)}`, async () => {
    let calls = 0;
    assert.equal(await recoverEmptyForward(value, root(RESOURCE), () => { calls++; return FULL; }), value);
    assert.equal(calls, 0);
  });

for (const id of ['', null, undefined, 10001, '123456', 'https://fixture.invalid/body', '../other', '/local/path', 'x'.repeat(513)])
  test(`缺失/无效资源不猜测其他消息、URL或路径：${String(id).slice(0,40)}`, async () => {
    let calls = 0;
    assert.equal(await recoverEmptyForward(EMPTY, root(id), () => { calls++; return FULL; }), EMPTY);
    assert.equal(calls, 0);
  });

test('资源ID只取原始元素，忽略顶层和外部自报的备用字段', async () => {
  let calls = 0;
  assert.equal(await recoverEmptyForward(EMPTY, { resId: RESOURCE, url: 'https://fixture.invalid/', elements: [] }, () => { calls++; return FULL; }), EMPTY);
  assert.equal(calls, 0);
});

test('同一消息含多个转发资源时不猜测应读取哪一个', async () => {
  const message = root(RESOURCE); message.elements.push(...root('other_resource').elements);
  let calls = 0;
  assert.equal(await recoverEmptyForward(EMPTY, message, () => { calls++; return FULL; }), EMPTY);
  assert.equal(calls, 0);
});

for (const value of [EMPTY, null, {}, { messages: {} }, { messages: undefined }])
  test(`补读没有正文时保持原不可读结果：${JSON.stringify(value)}`, async () => {
    let calls = 0;
    assert.equal(await recoverEmptyForward(EMPTY, root(RESOURCE), () => { calls++; return value; }), EMPTY);
    assert.equal(calls, 1);
  });

test('资源读取器抛错不泄漏异常正文或伪造成功', async () => {
  assert.equal(await recoverEmptyForward(EMPTY, root(RESOURCE), () => { throw new Error('fixture-private-detail'); }), EMPTY);
});

test('各次调用独享来源，不跨账号或会话缓存资源/正文', async () => {
  const calls = [];
  await recoverEmptyForward(EMPTY, root(RESOURCE), id => { calls.push(['a', id]); return FULL; });
  await recoverEmptyForward(EMPTY, root('fixture_resource_B'), id => { calls.push(['b', id]); return EMPTY; });
  assert.deepEqual(calls, [['a', RESOURCE], ['b', 'fixture_resource_B']]);
});

for (const budgetMs of [0, -1, NaN, Infinity]) test(`已耗尽或无效预算不续读：${budgetMs}`, async () => {
  let calls = 0;
  assert.equal(await recoverEmptyForward(EMPTY, root(RESOURCE), () => { calls++; return FULL; }, { budgetMs }), EMPTY);
  assert.equal(calls, 0);
});

test('底层读取超时后只保留原结果，迟到结果不触发续读', async () => {
  let release; let calls = 0;
  const result = recoverEmptyForward(EMPTY, root(RESOURCE), () => {
    calls++; return new Promise(resolve => { release = resolve; });
  }, { budgetMs: 5 });
  assert.equal(await result, EMPTY); release(FULL);
  await new Promise(resolve => setImmediate(resolve)); assert.equal(calls, 1);
});

test('未核验的NapCat构建必须拒绝，不按相似函数名模糊打补丁', () => {
  assert.throws(() => rewriteNapCatBundle('class Fixture { async _handle() {} }'), /source_not_verified/);
});
