"""Subprocess sandbox: reads JSON from stdin, executes plugin.run(), writes JSON to stdout.

Bytecode caching is intentionally disabled (``sys.dont_write_bytecode = True``)
so that every subprocess invocation always compiles the source file from scratch.
This prevents stale ``.pyc`` files from being loaded when a plugin is updated
within the same filesystem-timestamp second.

Usage (internal):
    echo '{"plugin_path": "/path/to/plugin.py", "input_data": {...}}' | python -m core.sandbox_wrapper
"""

# ⚠️  Must be set before any user import so Python never writes .pyc files
# inside this subprocess.  This eliminates stale-bytecode bugs when a plugin
# source file is updated faster than the OS filesystem mtime resolution.
import sys
sys.dont_write_bytecode = True

import importlib.util  # noqa: E402
import json            # noqa: E402
import traceback       # noqa: E402
import uuid            # noqa: E402


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        plugin_path: str = payload["plugin_path"]
        input_data: dict = payload["input_data"]

        # Use a unique module name per invocation to prevent any accidental
        # collision with previously cached module objects in sys.modules.
        # uuid4 guarantees global uniqueness; the 32-char hex suffix is
        # ephemeral (never registered in sys.modules after the process exits),
        # so there is no long-term pollution concern.
        module_name = f"_sandbox_plugin_{uuid.uuid4().hex}"

        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
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
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception:  # noqa: BLE001
        sys.stdout.write(json.dumps({"error": traceback.format_exc()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
