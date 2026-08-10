# file.kiwi 数据集下载流程记录

本文记录之前在 4090D 机器上从 file.kiwi 下载 `libero_combined` zstd 数据集的流程，方便后续复用。

## 背景

原始分享链接：

```text
https://file.kiwi/<room-id>#<fragment-key>
```

关键参数：

| 字段 | 值 |
|---|---|
| room id | `83bb9c11` |
| fragment key | `<fragment-key>` |
| file id | `502a3cb3` |
| sname | `cf` |
| dd | `0524` |
| chunks | `1428` |
| chunk size | `20,971,520` bytes |
| final archive size | `29,940,054,849` bytes |
| archive name | `libero_combined_filekiwi.tar.zst` |

file.kiwi 不是普通静态下载链接。它会把文件拆成多个加密 chunk，页面里的下载流程大致是：

1. 用 Firebase anonymous auth 获取临时 token。
2. 调用 `https://file.kiwi/api/getdlurl` 获取每批 chunk 的 Cloudflare R2 signed URL。
3. 下载每个 chunk。
4. 用 URL fragment key 做 HKDF/AES-GCM 解密。
5. 按 chunk offset 拼回最终 `.tar.zst` 文件。
6. `zstd -t` 校验后解压。

## 文件位置约定

4090D 上当时使用的路径：

```bash
BASE=/data/lerobot-imf-attnres-exp
ARCHIVE=$BASE/archives/libero_combined_filekiwi.tar.zst
PARTS=$ARCHIVE.parts
MANIFEST=$BASE/archives/filekiwi_manifest.json
TARGET=$BASE/datasets/libero_combined
```

本地临时脚本曾放在：

```bash
/tmp/make_filekiwi_manifest.mjs
/tmp/download_filekiwi_libero.mjs
/tmp/run_filekiwi_download_extract_train.sh
```

正式复用时建议放到：

```bash
/data/lerobot-imf-attnres-exp/scripts/make_filekiwi_manifest.mjs
/data/lerobot-imf-attnres-exp/scripts/download_filekiwi_libero.mjs
/data/lerobot-imf-attnres-exp/scripts/run_filekiwi_download_extract.sh
```

## Step 1：生成 manifest

如果远端机器可以访问 Google/Firebase，可以直接在远端生成 manifest；否则先在本地生成，再传到远端。

manifest 生成脚本逻辑：

1. 调用 Firebase anonymous sign up：

```text
https://identitytoolkit.googleapis.com/v1/accounts:signUp
```

2. 用拿到的 `idToken` 调用：

```text
https://file.kiwi/api/getdlurl
```

请求体形如：

```json
{
  "sname": "cf",
  "chunks": 1428,
  "chunk": 0,
  "id": "83bb9c11",
  "idf": "502a3cb3",
  "dd": "0524"
}
```

每 100 个 chunk 请求一次，得到 `head`、`dntail`、`dlist`，最终保存为：

```bash
/data/lerobot-imf-attnres-exp/archives/filekiwi_manifest.json
```

当时生成日志类似：

```text
[manifest] start=0 count=100
[manifest] start=100 count=100
...
[manifest] start=1400 count=28
[manifest] wrote /tmp/filekiwi_manifest.json
```

## Step 2：远端下载并解密 chunks

下载命令示例：

```bash
BASE=/data/lerobot-imf-attnres-exp
mkdir -p "$BASE/archives"

CONCURRENCY=8 node "$BASE/scripts/download_filekiwi_libero.mjs" \
  --manifest "$BASE/archives/filekiwi_manifest.json" \
  --output "$BASE/archives/libero_combined_filekiwi.tar.zst" \
  --parts-dir "$BASE/archives/libero_combined_filekiwi.tar.zst.parts" \
  --concurrency 8
```

说明：

- `--manifest`：提前生成的 signed URL 信息。
- `--output`：最终拼接出的 `.tar.zst` 文件。
- `--parts-dir`：每个 chunk 下载/解密完成后写一个 `.done` 标记，支持断点续传。
- `--concurrency`：并发下载数，之前用 `8`。网络稳定时可以调大，失败多时调小。

下载脚本核心逻辑：

1. 根据 chunk index 找到对应 batch。
2. 拼出 Cloudflare R2 signed URL：

```text
{head}/{chunk_id}?X-Amz-Signature={sig}&{dntail}
```

3. 下载加密 chunk。
4. 用下载页面提供的 fragment key（不要把真实 key 写入仓库）：

```text
<fragment-key>
```

做 `HKDF-SHA256` 派生 AES key 和 nonce。

5. 用 `AES-GCM-128` 解密 chunk。
6. 按 `idx * chunk_size` 写入最终 archive。
7. 写入 `.done` marker，失败后可恢复。

## Step 3：校验 archive

下载完成后先校验 zstd 文件：

```bash
zstd -t /data/lerobot-imf-attnres-exp/archives/libero_combined_filekiwi.tar.zst
```

只有 `zstd -t` 成功后再解压。

## Step 4：解压到数据集目录

```bash
BASE=/data/lerobot-imf-attnres-exp
ARCHIVE=$BASE/archives/libero_combined_filekiwi.tar.zst
DATASETS=$BASE/datasets
TARGET=$DATASETS/libero_combined

mkdir -p "$DATASETS"
TMP="$DATASETS/.extract_libero_combined_$(date +%s)"
mkdir -p "$TMP"

tar --use-compress-program=zstd -xf "$ARCHIVE" -C "$TMP"

if [ -d "$TMP/libero_combined" ]; then
  mv "$TMP/libero_combined" "$TARGET"
elif [ -d "$TMP/libero" ]; then
  mv "$TMP/libero" "$TARGET"
elif [ -f "$TMP/meta/info.json" ]; then
  mkdir -p "$TARGET"
  shopt -s dotglob
  mv "$TMP"/* "$TARGET"/
  shopt -u dotglob
else
  echo "ERROR: unknown archive layout" >&2
  find "$TMP" -maxdepth 2 -type f | head -n 40 >&2
  exit 3
fi

rmdir "$TMP" 2>/dev/null || true
```

## Step 5：检查数据集

```bash
python3 - <<'PY'
import json
p = '/data/lerobot-imf-attnres-exp/datasets/libero_combined/meta/info.json'
with open(p) as f:
    j = json.load(f)
print('path', p)
for k in ['repo_id', 'total_episodes', 'total_frames', 'fps', 'video']:
    print(k, j.get(k))
print('tasks', len(j.get('tasks', [])) if isinstance(j.get('tasks'), list) else j.get('total_tasks'))
PY
```

## 一键 orchestration 脚本示例

```bash
#!/usr/bin/env bash
set -euo pipefail

BASE=/data/lerobot-imf-attnres-exp
ARCHIVE="$BASE/archives/libero_combined_filekiwi.tar.zst"
PARTS="$ARCHIVE.parts"
DATASETS="$BASE/datasets"
TARGET="$DATASETS/libero_combined"
MANIFEST="$BASE/archives/filekiwi_manifest.json"
NODE_SCRIPT="$BASE/scripts/download_filekiwi_libero.mjs"

mkdir -p "$BASE/archives" "$DATASETS"

echo "[orchestrator] $(date -Is) start"
CONCURRENCY="${CONCURRENCY:-8}" node "$NODE_SCRIPT" \
  --manifest "$MANIFEST" \
  --output "$ARCHIVE" \
  --parts-dir "$PARTS" \
  --concurrency "$CONCURRENCY"

echo "[orchestrator] $(date -Is) zstd test"
zstd -t "$ARCHIVE"

echo "[orchestrator] $(date -Is) extract"
if [ -d "$TARGET" ] && [ -f "$TARGET/meta/info.json" ]; then
  echo "[orchestrator] dataset already exists: $TARGET"
elif [ -e "$TARGET" ]; then
  echo "[orchestrator] ERROR: target exists but meta/info.json missing: $TARGET" >&2
  exit 2
else
  TMP="$DATASETS/.extract_libero_combined_$(date +%s)"
  mkdir -p "$TMP"
  tar --use-compress-program=zstd -xf "$ARCHIVE" -C "$TMP"
  if [ -d "$TMP/libero_combined" ]; then
    mv "$TMP/libero_combined" "$TARGET"
  elif [ -d "$TMP/libero" ]; then
    mv "$TMP/libero" "$TARGET"
  elif [ -f "$TMP/meta/info.json" ]; then
    mkdir -p "$TARGET"
    shopt -s dotglob
    mv "$TMP"/* "$TARGET"/
    shopt -u dotglob
  else
    echo "[orchestrator] ERROR: unknown archive layout" >&2
    find "$TMP" -maxdepth 2 -type f | head -n 40 >&2
    exit 3
  fi
  rmdir "$TMP" 2>/dev/null || true
fi

echo "[orchestrator] $(date -Is) done"
```

## 常见问题

### 远端不能访问 Google/Firebase

症状：`identitytoolkit.googleapis.com` 请求失败，无法生成 token。

解决：在本地生成 `filekiwi_manifest.json`，再传到远端。远端只需要访问 Cloudflare R2 signed URL。

### signed URL 过期

症状：chunk 下载返回 `403` 或 `404`。

解决：重新生成 manifest，或者让下载脚本重新调用 `getdlurl` 获取新的 batch。

### 下载中断

脚本会在 `parts-dir` 下写 `.done` marker。重新运行同一命令会跳过已完成 chunk。

### zstd 校验失败

不要解压。优先检查：

1. 是否所有 chunk 都有 `.done` marker。
2. 输出文件大小是否等于 `29,940,054,849` bytes。
3. 是否有 chunk 解密 size mismatch 日志。
4. 重新跑下载脚本补齐/覆盖失败 chunk。

### archive layout 不确定

解压后可能出现：

- `libero_combined/`
- `libero/`
- 直接包含 `meta/info.json`

上面的解压脚本兼容这三种布局。
