"""
§7 @tool —— 把普通 Python 函数变成带 JSON Schema 的工具（对齐 LangChain 的 @tool）

模型调用工具时需要一份 JSON Schema（名字 / 描述 / 参数），手写又臭又长，
而且和函数签名容易脱节。@tool 从**签名 + 类型注解 + docstring**自动生成：

    from minigraph.tools import tool

    @tool
    def get_weather(city: str, unit: str = "C") -> str:
        '''查询城市天气（演示数据）。

        Args:
            city: 城市名，如 北京、上海
            unit: 温度单位，C 或 F
        '''
        return f"{city}: 晴 18°C"

    get_weather.schema()
    # {'name': 'get_weather',
    #  'description': '查询城市天气（演示数据）。',
    #  'input_schema': {'type': 'object',
    #                   'properties': {'city': {'type': 'string', 'description': '城市名，如 北京、上海'},
    #                                  'unit': {'type': 'string', 'description': '温度单位，C 或 F'}},
    #                   'required': ['city']}}          # 有默认值的参数不进 required

    get_weather.invoke({"city": "北京"})   # 等价于 get_weather(city="北京")

两种装饰写法都行：`@tool` 或 `@tool(name="...", description="...")`。
生成的 schema 是 **Anthropic tool 格式**（name/description/input_schema），
也正是 OpenAI 等兼容端点 `tools` 参数所要求结构的超集子集对应物 ——
minigraph 的 agent 示例统一走 Anthropic Messages 协议，所以用它。

解析规则（刻意保持简单，教学可读完）：
  - 参数类型：str/int/float/bool/list/dict -> string/integer/number/boolean/array/object；
    带注解的容器（list[str]、dict[str, int]）按容器映射；Optional[X] 取 X；
    没注解或认不出的类型一律当 string。
  - required：没有默认值的参数。
  - 描述：docstring 里 Args:/参数: 之前的部分；Args 段里 `name: 描述`
    （Google 风格，`name (type): 描述` 也认）填进对应参数的 description。
"""

from __future__ import annotations

import inspect
import re
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

_TYPE_MAP = {
    str: "string", int: "integer", float: "number", bool: "boolean",
    list: "array", dict: "object",
}


def _schema_type(tp: Any) -> str:
    if tp in _TYPE_MAP:
        return _TYPE_MAP[tp]
    origin = typing.get_origin(tp)
    if origin in (list, tuple, set):
        return "array"
    if origin is dict:
        return "object"
    if origin in (typing.Union, types.UnionType):   # Optional[X] / X | None
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return _schema_type(args[0])
    return "string"


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """拆出 (工具描述, {参数名: 参数描述})。"""
    if not doc:
        return "", {}
    desc_lines: list[str] = []
    params: dict[str, str] = {}
    in_args = False
    for raw in inspect.cleandoc(doc).splitlines():
        line = raw.strip()
        if line.rstrip(":") in ("Args", "Arguments", "Parameters", "参数"):
            in_args = True
            continue
        if in_args:
            m = re.match(r"^([A-Za-z_]\w*)\s*(\([^)]*\))?\s*[:：]\s*(.+)$", line)
            if m:
                params[m.group(1)] = m.group(3).strip()
        elif line:
            desc_lines.append(line)
    return "\n".join(desc_lines).strip(), params


@dataclass
class Tool:
    """一个工具 = 可调用函数 + 给模型看的 JSON Schema。"""

    name: str
    description: str
    input_schema: dict
    func: Callable = field(repr=False)

    def __call__(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)

    def invoke(self, args: dict | None = None) -> Any:
        """按模型给的 input dict 调用（LangChain 的 Tool.invoke 习惯）。"""
        return self.func(**(args or {}))

    def schema(self) -> dict:
        """Anthropic Messages API 的 tools 参数里的一项。"""
        return {"name": self.name, "description": self.description,
                "input_schema": self.input_schema}


def tool(fn: Callable | None = None, *, name: str | None = None,
         description: str | None = None):
    """
    @tool / @tool(name=..., description=...)：函数 -> Tool。

    被装饰后仍可直接调用（Tool.__call__ 委托原函数），只是额外带上了
    .name / .description / .input_schema / .schema() / .invoke()。
    """
    def deco(f: Callable) -> Tool:
        hints = get_type_hints(f)
        doc_desc, param_desc = _parse_docstring(f.__doc__)
        properties: dict[str, dict] = {}
        required: list[str] = []
        for p in inspect.signature(f).parameters.values():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue                       # *args/**kwargs 不进 schema
            prop: dict[str, Any] = {"type": _schema_type(hints.get(p.name, str))}
            if p.name in param_desc:
                prop["description"] = param_desc[p.name]
            if p.default is not inspect.Parameter.empty:
                prop["default"] = p.default
            else:
                required.append(p.name)
            properties[p.name] = prop
        return Tool(
            name=name or f.__name__,
            description=description or doc_desc or f.__name__,
            input_schema={"type": "object", "properties": properties,
                          "required": required},
            func=f,
        )

    return deco if fn is None else deco(fn)
