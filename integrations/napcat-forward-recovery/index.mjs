/**
 * NapCat 4.18.37的空转发读取补偿。核心函数只接收原动作已取得的RawMessage及原资源读取器。
 * rewrite只生成本地候选，不写正式文件、不加载模块、不重启QQ、不访问任何网络或发送器。
 */
import { createHash } from 'node:crypto';

export const BASE_SHA256 = '96934cbd3a4a7ad4d50bfeeaa10adb600fd142cbb4f1be00ce427287fd5926fb';
export const PATCH_MARKER = 'shinku-empty-forward-recovery-v1';

export async function recoverEmptyForward(original, rawMessage, readResource, { budgetMs = 3500 } = {}) {
  // 正常结果、异常数据形状及没有可信资源ID的情况沿用原结果，不能猜测其他消息或会话。
  if (!Array.isArray(original?.messages) || original.messages.length || typeof readResource !== 'function') return original;
  const forwards = Array.isArray(rawMessage?.elements) ? rawMessage.elements.filter(row => row?.multiForwardMsgElement) : [];
  if (forwards.length !== 1) return original;
  const resource = forwards[0].multiForwardMsgElement.resId;
  if (typeof resource !== 'string' || !/^[A-Za-z0-9_+\-][A-Za-z0-9_+=/.-]{0,511}$/.test(resource)
      || /^\d+$/.test(resource) || resource.split('/').includes('..')
      || !Number.isFinite(budgetMs) || budgetMs <= 0) return original;
  let timer;
  try {
    // readResource由原动作闭包提供；只读本条RawMessage自带resId，不请求卡片URL或本地路径。
    // QQ底层Promise不能取消，超时后不再续读；每次动作最多补一次，不能把空结果伪装成成功。
    const result = await Promise.race([
      Promise.resolve().then(() => readResource(resource)),
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('forward_resource_read_timeout')), Math.min(3500, budgetMs)); }),
    ]);
    return Array.isArray(result?.messages) && result.messages.length > 0 ? result : original;
  } catch {
    // 保留原不可读状态，不向客户端泄漏异常正文、凭据或原始聊天内容。
    return original;
  } finally { clearTimeout(timer); }
}

export function rewriteNapCatBundle(source) {
  // 构建文件的变量名会随版本变化；仅接受已独立核对的原始字节，避免给其他版本错误打补丁。
  if (createHash('sha256').update(source).digest('hex') !== BASE_SHA256)
    throw new Error('napcat_forward_source_not_verified');
  const declaration = /class (\w+) extends ue \{\n  actionName = re\.GoCQHTTP_GetForwardMsg;/g;
  const matches = [...source.matchAll(declaration)];
  if (matches.length !== 1) throw new Error('napcat_forward_action_not_unique');
  const start = matches[0].index;
  const end = source.indexOf('\n}\n', start) + 3;
  if (end < start) throw new Error('napcat_forward_action_incomplete');
  const action = source.slice(start, end);
  const entry = '  async _handle(e) {\n';
  const original = '          const d = l.data.content;\n          if (d)\n            return { messages: d };';
  if (action.split(entry).length !== 2 || action.split(original).length !== 2)
    throw new Error('napcat_forward_patch_site_not_unique');
  const replacement = [
    '          const d = l.data.content;',
    '          if (Array.isArray(d) && d.length === 0)',
    '            return await __shinkuRecoverEmptyForward({ messages: d }, c, i, { budgetMs: 4000 - (performance.now() - __shinkuForwardStarted) });',
    '          if (d)',
    '            return { messages: d };',
  ].join('\n');
  const patchedAction = action.replace(entry, entry + '    const __shinkuForwardStarted = performance.now();\n').replace(original, replacement);
  const helper = `// ${PATCH_MARKER}：仅补偿同一原消息的空转发结果；来源、作者与协议解析继续由原动作负责。\n`
    + `const __shinkuRecoverEmptyForward = ${recoverEmptyForward.toString()};\n`;
  const output = source.slice(0, start) + helper + patchedAction + source.slice(end);
  return { source: output, actionClass: matches[0][1], sha256: createHash('sha256').update(output).digest('hex') };
}
