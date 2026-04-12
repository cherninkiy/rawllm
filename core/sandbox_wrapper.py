"""Subprocess sandbox: reads JSON from stdin, executes plugin.run(), writes JSON to stdout.

Usage (internal):
    echo '{"plugin_path": "/path/to/plugin.py", "input_data": {...}}' | python -m core.sandbox_wrapper
"""

import importlib.util
import json
import sys
import traceback


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        plugin_path: str = payload["plugin_path"]
        input_data: dict = payload["input_data"]

        spec = importlib.util.spec_from_file_location("_sandbox_plugin", plugin_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {plugin_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        # Mirror in-process behaviour: call init() if it takes no required args.
        init_fn = getattr(module, "init", None)
        if init_fn is not None:
            try:
                init_fn()
            except TypeError:
                pass  # init requires arguments – skip
            except Exception:
                pass  # non-fatal

        run_fn = getattr(module, "run", None)
        if run_fn is None:
            raise AttributeError(f"Plugin at {plugin_path!r} has no run() function.")

        result = run_fn(input_data)
        sys.stdout.write(json.dumps({"result": result}, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        sys.stdout.write(json.dumps({"error": traceback.format_exc()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
