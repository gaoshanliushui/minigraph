"""@tool：函数签名 + docstring -> 工具 JSON Schema。"""

from typing import Optional

from minigraph.tools import Tool, tool


@tool
def get_weather(city: str, unit: str = "C", days: int = 1) -> str:
    """查询城市天气。

    Args:
        city: 城市名，如 北京
        unit: 温度单位，C 或 F
    """
    return f"{city}: 晴"


def test_schema_from_signature_and_docstring():
    s = get_weather.schema()
    assert s["name"] == "get_weather"
    assert s["description"] == "查询城市天气。"
    props = s["input_schema"]["properties"]
    assert props["city"] == {"type": "string", "description": "城市名，如 北京"}
    assert props["unit"] == {"type": "string", "description": "温度单位，C 或 F",
                             "default": "C"}
    assert props["days"] == {"type": "integer", "default": 1}   # 无参数描述
    # 有默认值的参数不进 required
    assert s["input_schema"]["required"] == ["city"]
    assert s["input_schema"]["type"] == "object"


def test_tool_is_still_callable_and_invokable():
    assert get_weather("北京") == "北京: 晴"
    assert get_weather.invoke({"city": "上海", "unit": "F"}) == "上海: 晴"


def test_decorator_with_overrides():
    @tool(name="wx", description="天气")
    def f(city: str) -> str:
        return city

    assert isinstance(f, Tool)
    assert f.name == "wx"
    assert f.schema()["description"] == "天气"
    assert f("北京") == "北京"


def test_type_mapping_and_optional():
    @tool
    def f(a: float, b: bool, c: list[str], d: dict, e: Optional[int] = None, g=None):
        ...

    props = f.input_schema["properties"]
    assert props["a"]["type"] == "number"
    assert props["b"]["type"] == "boolean"
    assert props["c"]["type"] == "array"
    assert props["d"]["type"] == "object"
    assert props["e"]["type"] == "integer"       # Optional[int] -> integer
    assert props["g"]["type"] == "string"        # 无注解默认 string
    assert f.input_schema["required"] == ["a", "b", "c", "d"]   # e/g 有默认值


def test_google_style_param_with_type_and_no_docstring():
    @tool
    def f(x: int) -> int:
        """算个东西。

        Args:
            x (int): 输入值
        """
        return x

    assert f.input_schema["properties"]["x"]["description"] == "输入值"

    @tool
    def noop() -> None:
        ...

    assert noop.description == "noop"            # 没 docstring 就退回函数名
    assert noop.input_schema["properties"] == {}
    assert noop.input_schema["required"] == []
