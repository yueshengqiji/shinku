# NapCat 空转发读取补偿

此目录保存实际恢复函数与精确构建重写器，不是NapCat插件。当前仅核验NapCat 4.18.37，
原napcat.mjs的SHA256必须等于index.mjs中的BASE_SHA256；其他构建和重复处理均拒绝。
不改变官方插件白名单，不安装依赖，不替换整个NapCat发行包。

原get_forward_msg动作先按账号内消息ID找到原RawMessage，再解析转发内容。
有正文的正常路径保持原样。仅在得到空数组时，从该RawMessage唯一的multiForwardMsgElement
取得resId，调用原动作已有的资源读取闭包一次。没有资源、存在歧义、补读失败或超时都保留
原不可读结果；没有跨会话缓存、身份补造或任意URL/文件读取。

补读最多3.5秒，且不得延伸到原动作开始后4秒之外。QQ底层Promise无法取消，超时后的结果
被丢弃且不再续读。原动作自己的超时和错误行为保持。候选仅增加恢复函数、起始计时和空数组
分支；原消息ID解析、会话定位、作者转换、权限和其他动作不变。

公共离线用例：

```bash
python scripts/run_desktop_batch.py -- node --test tests/test_napcat_forward_recovery.mjs
```

重写函数rewriteNapCatBundle(source)返回候选source和sha256；不写正式文件或启动程序。
完整候选需先语法检查，并针对已安装NapCat的真实动作做隔离差分。实际聊天数据、构建副本、
加载脚本、备份与回放结果只保存在本机private/维护目录，不进入此公共源码目录。

正式加载需替换容器中的单个napcat.mjs并重启该NapCat容器，不能用Console重启代替。
未取得这项运行操作授权前只构造候选。加载后须只读回放原失败记录，并核对新记录、登录和
配置状态；失败只回退本次构建，不恢复数据库或重放事件。容器重建/发行版更新可能移除此
本地增量，后续必须重新核验版本和源码，不能把旧重写器盲目套到新构建。
