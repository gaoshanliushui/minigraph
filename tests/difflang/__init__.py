"""
差分测试：把同一张图分别跑在真实 LangGraph 和 minigraph 上，逐项比对。

为什么要做这个？
    minigraph 是"教学实现"，它宣称对齐 LangGraph 的内核语义。
    光靠自己写断言，只能证明"实现符合我的理解"，不能证明"我的理解是对的"。
    唯一可靠的办法是拿真实 LangGraph 当基准跑一遍。

怎么跑：
    解释器必须同时能 import langgraph 和 minigraph。
    minigraph 是纯标准库实现，所以用装了 langgraph 的那个解释器，
    把仓库根目录塞进 sys.path 即可。

    /tmp/lgvenv/bin/python -m pytest tests/test_diff_langgraph.py -v

    （没装 langgraph 时整个文件 skip，不影响普通测试。）
"""
