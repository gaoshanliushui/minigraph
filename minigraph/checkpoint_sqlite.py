"""
§3b SQLite checkpointer

把 checkpoint 落盘到 SQLite。接口与 `InMemorySaver` **完全一致**，
所以换个 saver，就能把"只在内存里"的图变成"重启进程还能接着跑"的图：

    from minigraph.checkpoint_sqlite import SqliteSaver

    saver = SqliteSaver("checkpoints.db")
    app = g.compile(checkpointer=saver)

对照真实 LangGraph：`langgraph/checkpoint/sqlite/__init__.py` 的 `SqliteSaver`
（同样的表结构思路：thread_id + checkpoint_id + 一列序列化后的快照）。

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
import uuid
from typing import Any

from .checkpoint import Checkpoint
from .constants import MISSING
from .control import Send

# 类型标签的键名。刻意起个不容易撞的名字 —— 用户的 state 里可能真的有个
# 叫 "__type__" 的字段。
_TAG = "__mg_type__"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    seq           INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id     TEXT    NOT NULL,
    checkpoint_id TEXT    NOT NULL,
    step          INTEGER NOT NULL,
    data          BLOB    NOT NULL,
    created_at    REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread
    ON checkpoints (thread_id, seq);
"""

# Checkpoint 的全部字段。序列化时按这个顺序取，反序列化时按它还原 ——
# 显式列出，避免以后给 Checkpoint 加字段时忘了同步这里。
_CKPT_FIELDS = (
    "id", "step", "values", "next", "tasks", "interrupt", "interrupts",
    "resumes", "slots", "channel_values", "channel_versions", "versions_seen",
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
        f"以及 minigraph 的 Send / MISSING"
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
        self.lock = threading.Lock()
        self._n = 0
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
            self.conn.commit()

    # -- InMemorySaver 同名接口 --------------------------------------------
    def new_id(self) -> str:
        """生成 checkpoint id。带随机后缀 —— 多个进程写同一个库也不会撞。"""
        with self.lock:
            self._n += 1
            return f"ckpt-{self._n:04d}-{uuid.uuid4().hex[:8]}"

    def put(self, thread_id: str, ckpt: Checkpoint) -> str:
        blob = json.dumps(_encode(_ckpt_to_dict(ckpt)),
                          ensure_ascii=False).encode("utf-8")
        with self.lock:
            self.conn.execute(
                "INSERT INTO checkpoints"
                " (thread_id, checkpoint_id, step, data, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (thread_id, ckpt.id, ckpt.step, blob, time.time()),
            )
            self.conn.commit()
        return ckpt.id

    def list(self, thread_id: str) -> list[Checkpoint]:
        """该 thread 的全部 checkpoint，按写入顺序（旧 -> 新）。"""
        with self.lock:
            rows = self.conn.execute(
                "SELECT data FROM checkpoints WHERE thread_id = ? ORDER BY seq",
                (thread_id,),
            ).fetchall()
        return [_ckpt_from_dict(_decode(json.loads(r[0]))) for r in rows]

    def get(self, thread_id: str, checkpoint_id: str | None = None) -> Checkpoint | None:
        """取某个 checkpoint；不给 id 就取最新的那个。"""
        with self.lock:
            if checkpoint_id is None:
                row = self.conn.execute(
                    "SELECT data FROM checkpoints WHERE thread_id = ?"
                    " ORDER BY seq DESC LIMIT 1",
                    (thread_id,),
                ).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT data FROM checkpoints"
                    " WHERE thread_id = ? AND checkpoint_id = ?",
                    (thread_id, checkpoint_id),
                ).fetchone()
        if row is None:
            if checkpoint_id is None:
                return None
            raise KeyError(f"thread {thread_id} 下找不到 checkpoint {checkpoint_id}")
        return _ckpt_from_dict(_decode(json.loads(row[0])))

    # -- 额外的好用接口 ----------------------------------------------------
    def threads(self) -> list[str]:
        """所有有 checkpoint 的 thread_id。"""
        with self.lock:
            rows = self.conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
            ).fetchall()
        return [r[0] for r in rows]

    def delete_thread(self, thread_id: str) -> None:
        """删掉一个 thread 的全部 checkpoint。"""
        with self.lock:
            self.conn.execute("DELETE FROM checkpoints WHERE thread_id = ?",
                              (thread_id,))
            self.conn.commit()

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def __enter__(self) -> "SqliteSaver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
