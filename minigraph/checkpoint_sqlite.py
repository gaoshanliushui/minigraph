"""
§3b SQLite checkpointer

把 checkpoint 落盘到 SQLite。接口与 `InMemorySaver` **完全一致**，
所以换个 saver，就能把"只在内存里"的图变成"重启进程还能接着跑"的图：

    from minigraph.checkpoint_sqlite import SqliteSaver

    saver = SqliteSaver("checkpoints.db")
    app = g.compile(checkpointer=saver)

对照真实 LangGraph：`langgraph/checkpoint/sqlite/__init__.py` 的 `SqliteSaver`
（同样的两表结构：`checkpoints` 存快照 blob + `parent_checkpoint_id`，
`writes` 存增量写入日志）。

---

## 存储布局

```
checkpoints(thread_id, checkpoint_id, parent_checkpoint_id, step, data, ...)
    data = 状态/版本/水位线等内核状态的 JSON blob（不含写入日志）
writes(thread_id, checkpoint_id, idx, channel, value)
    每份 checkpoint 当步的增量写入，idx 保序
```

`Checkpoint.pending_writes` 因此是个**查询视图**：写入时拆进 writes 表，
读出时拼回。`parent_checkpoint_id` 支撑显式祖先链 —— 分叉后存储顺序
不等于祖先后代关系，delta 重放必须沿父链走。

旧格式的库打开时自动迁移：补 parent 列并按 seq 回填（旧历史是线性的），
旧的 blob 内 pending_writes 在 writes 表查不到时回退读取，不用搬数据。

---

## 存什么

一个 Checkpoint 里混着"能直接进 JSON 的"和"进不去的"东西：

    values / channel_versions / versions_seen / interrupts / slots   -> JSON 原生
    next            是 tuple（JSON 只会存成 list）
    tasks           可能是 Send 对象（带 .node / .arg）
    channel_values  可能是 MISSING 哨兵、set（屏障的 seen）、bytes
    resumes         用户给的 resume 值，什么都有可能

所以用**带类型标签的 JSON**：遇到 JSON 装不下的类型就包一层
`{"__mg_type__": "...", ...}`，读回来时按标签还原。

不用 pickle 的原因：反序列化不可信文件等于执行任意代码；而且 pickle
跨 Python 版本不稳。代价是自定义类型的 state 值存不进去（会明确报错，
而不是悄悄丢数据）。
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import threading
import time
from typing import Any

from .checkpoint import Checkpoint, uuid6
from .channels import DeltaSnapshot
from .constants import MISSING, WRITES_IDX_MAP
from .control import Send

# 类型标签的键名。刻意起个不容易撞的名字 —— 用户的 state 里可能真的有个
# 叫 "__type__" 的字段。
_TAG = "__mg_type__"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    seq           INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id     TEXT    NOT NULL,
    checkpoint_id TEXT    NOT NULL,
    parent_checkpoint_id TEXT,
    step          INTEGER NOT NULL,
    data          BLOB    NOT NULL,
    created_at    REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread
    ON checkpoints (thread_id, seq);
-- 增量写入单独一张表（对齐 LangGraph 的 writes 表），不塞进 checkpoint
-- 的 JSON blob：
--   - checkpoint blob 只装「状态 + 版本 + 水位线」这类结构化内核状态；
--   - 写入日志按 (checkpoint_id, idx) 有序存放，重放时按通道精确捞取，
--     不用反序列化整条历史的大 blob；
--   - parent_checkpoint_id 让祖先链可以显式地走（分叉后存储顺序 ≠
--     祖先后代关系）。
CREATE TABLE IF NOT EXISTS writes (
    thread_id     TEXT    NOT NULL,
    checkpoint_id TEXT    NOT NULL,
    idx           INTEGER NOT NULL,
    channel       TEXT    NOT NULL,
    value         BLOB    NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_id, idx)
);
"""

# Checkpoint 里**进 data blob** 的字段。序列化时按这个顺序取，反序列化时
# 按它还原 —— 显式列出，避免以后给 Checkpoint 加字段时忘了同步这里。
#
# 注意 pending_writes **不在**这里：它存 writes 表（见 _SCHEMA），读出来
# 时拼回 Checkpoint，所以对外 Checkpoint 仍是自包含的（pending_writes
# 从存储字段变成了查询视图，与 LangGraph 一致）。
# parent_id 是表列，也不进 blob。
_CKPT_FIELDS = (
    "id", "step", "values", "next", "tasks", "interrupt", "interrupts",
    "resumes", "slots", "channel_values", "channel_versions", "versions_seen",
    "delta_counters",
)


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------

def _encode(obj: Any) -> Any:
    """把对象编码成 JSON 能装下的结构；装不下的类型打类型标签。"""
    if obj is MISSING:
        return {_TAG: "missing"}
    if isinstance(obj, Send):
        return {_TAG: "send", "node": obj.node, "arg": _encode(obj.arg)}
    if isinstance(obj, DeltaSnapshot):
        return {_TAG: "delta_snapshot", "value": _encode(obj.value)}
    if isinstance(obj, (set, frozenset)):
        return {_TAG: "set", "items": [_encode(v) for v in obj]}
    if isinstance(obj, tuple):
        return {_TAG: "tuple", "items": [_encode(v) for v in obj]}
    if isinstance(obj, bytes):
        return {_TAG: "bytes", "b64": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj):
            return {k: _encode(v) for k, v in obj.items()}
        # 键不是字符串（用户 state 里可能有 {1: "a"}）—— 用带标签的形式保真
        return {_TAG: "dict", "items": [[_encode(k), _encode(v)]
                                        for k, v in obj.items()]}
    if isinstance(obj, list):
        return [_encode(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    raise TypeError(
        f"无法序列化 {type(obj).__name__}；"
        f"SQLite checkpointer 只支持 JSON 原生类型、tuple/set/bytes、"
        f"以及 minigraph 的 Send / MISSING / DeltaSnapshot"
    )


def _decode(obj: Any) -> Any:
    """_encode 的逆操作。"""
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    if isinstance(obj, dict):
        tag = obj.get(_TAG)
        keys = set(obj)
        # 只认"键完全吻合"的标签 —— 用户 state 里万一有个
        # {"__mg_type__": "tuple"} 这样的普通字典，不该被误当成标签。
        if tag == "missing" and keys == {_TAG}:
            return MISSING
        if tag == "send" and keys == {_TAG, "node", "arg"}:
            return Send(node=obj["node"], arg=_decode(obj["arg"]))
        if tag == "delta_snapshot" and keys == {_TAG, "value"}:
            return DeltaSnapshot(_decode(obj["value"]))
        if tag in ("set", "tuple") and keys == {_TAG, "items"}:
            items = [_decode(v) for v in obj["items"]]
            return set(items) if tag == "set" else tuple(items)
        if tag == "bytes" and keys == {_TAG, "b64"}:
            return base64.b64decode(obj["b64"])
        if tag == "dict" and keys == {_TAG, "items"}:
            return {_decode(k): _decode(v) for k, v in obj["items"]}
        return {k: _decode(v) for k, v in obj.items()}
    return obj


def _ckpt_to_dict(ckpt: Checkpoint) -> dict:
    return {f: getattr(ckpt, f) for f in _CKPT_FIELDS}


def _ckpt_from_dict(d: dict) -> Checkpoint:
    return Checkpoint(
        id=d["id"],
        step=d["step"],
        values=d["values"],
        next=tuple(d["next"]),
        tasks=list(d["tasks"]),
        interrupt=d["interrupt"],
        interrupts=list(d["interrupts"]),
        resumes={k: list(v) for k, v in d["resumes"].items()},
        slots=list(d["slots"]),
        channel_values=d["channel_values"],
        channel_versions=d["channel_versions"],
        versions_seen=d["versions_seen"],
        pending_writes=[(c, v) for c, v in d.get("pending_writes") or []],
        delta_counters={k: tuple(v) for k, v in
                        (d.get("delta_counters") or {}).items()},
    )


# ---------------------------------------------------------------------------
# Saver
# ---------------------------------------------------------------------------

class SqliteSaver:
    """
    把 checkpoint 存进 SQLite。接口与 `InMemorySaver` 一模一样，
    所以两者可以互换使用。

    Args:
        conn: 一个 `sqlite3.Connection`，或者数据库文件路径（`":memory:"`
            表示内存库）。传路径时自己建连接。

    线程安全：所有 SQL 都在一把锁里跑。连接用 `check_same_thread=False`
    建，这样节点线程里也能写 checkpoint。
    """

    def __init__(self, conn: sqlite3.Connection | str | os.PathLike = ":memory:",
                 *, timeout: float = 30.0) -> None:
        if isinstance(conn, (str, os.PathLike)):
            conn = sqlite3.connect(str(conn), check_same_thread=False, timeout=timeout)
        self.conn = conn
        # RLock：get_delta_channel_history 在持锁遍历父链时逐步调 get()
        # （get 自己也持锁），普通 Lock 会自死锁。
        self.lock = threading.RLock()
        self._setup()

    @classmethod
    def from_conn_string(cls, conn_string: str, *, timeout: float = 30.0) -> "SqliteSaver":
        """按连接串建 saver（`"checkpoints.db"` 或 `"file:mem.db?mode=memory"`）。"""
        return cls(conn_string, timeout=timeout)

    def _setup(self) -> None:
        with self.lock:
            # WAL：读写不互相阻塞（内存库会忽略它）
            try:
                self.conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:  # pragma: no cover - 某些文件系统不支持
                pass
            self.conn.executescript(_SCHEMA)
            self._migrate()
            self.conn.commit()

    def _migrate(self) -> None:
        """
        旧库升级：checkpoints 表补 parent_checkpoint_id 列。

        旧数据是**线性**历史（分叉支持是随父链一起引入的），所以按 seq
        顺序回填 parent 即可：每份的父节点 = 同 thread 里 seq 比它小的
        最近一份。各 thread 的第一份没有更早的 seq，子查询得 NULL，
        parent 保持 NULL（链的起点）—— 正好正确。

        旧数据的 pending_writes 还留在 data blob 里：读取时 writes 表
        查不到就回退用 blob 里的（见 _load_writes 的调用点），不用搬数据。
        """
        cols = {r[1] for r in
                self.conn.execute("PRAGMA table_info(checkpoints)")}
        if "parent_checkpoint_id" in cols:
            return
        self.conn.execute(
            "ALTER TABLE checkpoints ADD COLUMN parent_checkpoint_id TEXT")
        self.conn.execute(
            "UPDATE checkpoints SET parent_checkpoint_id = ("
            "  SELECT c2.checkpoint_id FROM checkpoints c2"
            "  WHERE c2.thread_id = checkpoints.thread_id"
            "    AND c2.seq < checkpoints.seq"
            "  ORDER BY c2.seq DESC LIMIT 1"
            ") WHERE parent_checkpoint_id IS NULL")

    # -- InMemorySaver 同名接口 --------------------------------------------
    def new_id(self) -> str:
        """
        UUIDv6：字典序 == 生成先后（多进程写同一个库也不会撞）。

        对齐 LangGraph：checkpoint id 单调递增，可直接当排序键；
        时间序高位在前也让 B-tree 索引总是末尾追加，局部性好。
        """
        return str(uuid6())

    def put(self, thread_id: str, ckpt: Checkpoint) -> str:
        blob = json.dumps(_encode(_ckpt_to_dict(ckpt)),
                          ensure_ascii=False).encode("utf-8")
        # 增量写入走 writes 表，不进 data blob。idx 保序 —— 同一超步内
        # 多条写入的先后顺序对顺序敏感的 reducer 有意义。
        # 特殊通道（__interrupt__/__resume__）按 WRITES_IDX_MAP 用负 idx，
        # 与普通写入（0 起）错开，互不冲突（对齐 LangGraph）。
        write_rows = [
            (thread_id, ckpt.id, WRITES_IDX_MAP.get(chan, idx), chan,
             json.dumps(_encode(val), ensure_ascii=False).encode("utf-8"))
            for idx, (chan, val) in enumerate(ckpt.pending_writes)
        ]
        with self.lock:
            self.conn.execute(
                "INSERT INTO checkpoints"
                " (thread_id, checkpoint_id, parent_checkpoint_id,"
                "  step, data, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (thread_id, ckpt.id, ckpt.parent_id, ckpt.step, blob,
                 time.time()),
            )
            self.conn.executemany(
                "INSERT OR REPLACE INTO writes"
                " (thread_id, checkpoint_id, idx, channel, value)"
                " VALUES (?, ?, ?, ?, ?)",
                write_rows,
            )
            self.conn.commit()
        return ckpt.id

    def _load_writes(self, thread_id: str,
                     checkpoint_id: str) -> list[tuple[str, Any]] | None:
        """
        从 writes 表读回某份 checkpoint 的增量写入（按 idx 保序）。

        返回 None 表示 writes 表里**没有**这个 checkpoint 的记录 ——
        调用方据此回退到旧格式（data blob 里的 pending_writes 字段）。
        """
        rows = self.conn.execute(
            "SELECT channel, value FROM writes"
            " WHERE thread_id = ? AND checkpoint_id = ? ORDER BY idx",
            (thread_id, checkpoint_id),
        ).fetchall()
        if not rows:
            return None
        return [(chan, _decode(json.loads(val))) for chan, val in rows]

    def _row_to_ckpt(self, thread_id: str, row) -> Checkpoint:
        """(parent_checkpoint_id, data) 行 -> Checkpoint（拼回写入视图）。"""
        parent_id, blob = row
        ckpt = _ckpt_from_dict(_decode(json.loads(blob)))
        ckpt.parent_id = parent_id
        writes = self._load_writes(thread_id, ckpt.id)
        if writes is not None:
            # 新格式：pending_writes 来自 writes 表。
            ckpt.pending_writes = writes
        # 否则是旧格式：_ckpt_from_dict 已经从 data blob 里读到了
        # pending_writes，原样保留即可。
        return ckpt

    def list(self, thread_id: str) -> list[Checkpoint]:
        """该 thread 的全部 checkpoint，按写入顺序（旧 -> 新）。"""
        with self.lock:
            rows = self.conn.execute(
                "SELECT parent_checkpoint_id, data FROM checkpoints"
                " WHERE thread_id = ? ORDER BY seq",
                (thread_id,),
            ).fetchall()
            return [self._row_to_ckpt(thread_id, r) for r in rows]

    def get(self, thread_id: str, checkpoint_id: str | None = None) -> Checkpoint | None:
        """取某个 checkpoint；不给 id 就取最新的那个。"""
        with self.lock:
            if checkpoint_id is None:
                row = self.conn.execute(
                    "SELECT parent_checkpoint_id, data FROM checkpoints"
                    " WHERE thread_id = ? ORDER BY seq DESC LIMIT 1",
                    (thread_id,),
                ).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT parent_checkpoint_id, data FROM checkpoints"
                    " WHERE thread_id = ? AND checkpoint_id = ?",
                    (thread_id, checkpoint_id),
                ).fetchone()
            if row is None:
                if checkpoint_id is None:
                    return None
                raise KeyError(
                    f"thread {thread_id} 下找不到 checkpoint {checkpoint_id}")
            return self._row_to_ckpt(thread_id, row)

    # -- 额外的好用接口 ----------------------------------------------------
    def threads(self) -> list[str]:
        """所有有 checkpoint 的 thread_id。"""
        with self.lock:
            rows = self.conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
            ).fetchall()
        return [r[0] for r in rows]

    def get_delta_channel_history(self, thread_id: str, channels: list[str],
                                  checkpoint_id: str | None = None
                                  ) -> dict[str, dict[str, Any]]:
        """
        沿 thread 的 **parent 链**往回走，取出每个 delta 通道的
        「种子 + 增量写入」。语义与 InMemorySaver 的同名方法一致。

        走法见 InMemorySaver.get_delta_channel_history 的 docstring：
        每个通道在**最近一个存过值的祖先**处停下，那份值就是 seed；
        seed 之后积累的写入就是 writes。

        分叉后必须走 parent_checkpoint_id，不能按 seq 顺序扫 ——
        存储里两条分叉线是交错追加的，按 seq 会把另一条线的写入
        也重放进来。

        （LangGraph 的 SqliteSaver 把这段走法写成了递归 CTE + 按通道
        UNION ALL 的两阶段 SQL，一次取齐；这里逐步 get 更好读，
        每步一次主键查询，深度 = 距种子的距离，通常很小。）
        """
        if not channels:
            return {}
        seed_by_ch: dict[str, Any] = {}
        writes_by_ch: dict[str, list[Any]] = {c: [] for c in channels}
        remaining = set(channels)

        # 先按新->旧收集 checkpoint，最后统一按旧->新展开写入 ——
        # 一个 checkpoint 里同一通道可能有多条写入，它们之间的顺序有意义。
        #
        # 关键：**找到种子的那份 checkpoint 不参与重放**。
        # 种子值 = 那个时刻的完整状态，已经包含了它（以及更早祖先）的写入；
        # 再重放一遍就重复了（计数器这种不幂等的 reducer 会直接算错）。
        collected = []
        walked: set[str] = set()          # 防御：parent 链成环时止步
        with self.lock:
            cursor = self.get(thread_id, checkpoint_id)
            while (cursor is not None and remaining
                   and cursor.id not in walked):
                walked.add(cursor.id)
                found_seed_here = False
                for ch in list(remaining):
                    if ch in cursor.channel_values:
                        seed_by_ch[ch] = cursor.channel_values[ch]
                        remaining.discard(ch)
                        found_seed_here = True
                if found_seed_here:
                    pass          # 种子自己的写入已在种子值里，跳过
                else:
                    collected.append(cursor)
                parent_id = cursor.parent_id
                cursor = self.get(thread_id, parent_id) if parent_id else None

        for ckpt in reversed(collected):
            for chan, val in ckpt.pending_writes:
                if chan in writes_by_ch:
                    writes_by_ch[chan].append(val)

        result: dict[str, dict[str, Any]] = {}
        for ch in channels:
            entry: dict[str, Any] = {"writes": writes_by_ch[ch]}
            if ch in seed_by_ch:
                entry["seed"] = seed_by_ch[ch]
            result[ch] = entry
        return result

    def delete_thread(self, thread_id: str) -> None:
        """删掉一个 thread 的全部 checkpoint 和写入日志。"""
        with self.lock:
            self.conn.execute("DELETE FROM checkpoints WHERE thread_id = ?",
                              (thread_id,))
            self.conn.execute("DELETE FROM writes WHERE thread_id = ?",
                              (thread_id,))
            self.conn.commit()

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def __enter__(self) -> "SqliteSaver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
