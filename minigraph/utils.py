
def print_graph(app):
    print("=== compiled channels (control-flow channels) ===")
    for k, v in app.channels.items():
        print(f"  {k!r:24} -> {type(v).__name__}")
    print()

    print("=== node : triggers / writers ===")
    for name, spec in app.nodes.items():
        print(f"  [{name}]")
        print("    triggers =", spec.triggers)
        print("    writers  =", spec.writers)

        if spec.branches:
            print("    branches =", [b["source"] for b in spec.branches])
        if spec.is_start:
            print("    (入口伪节点)")
        print()