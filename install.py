import os
import subprocess
import sys
import threading
import traceback

if sys.argv[0] == "install.py":
    sys.path.append(".")  # for portable version

impact_path = os.path.join(os.path.dirname(__file__), "modules")
comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

sys.path.append(impact_path)
sys.path.append(comfy_path)


def handle_stream(stream, is_stdout):
    for msg in stream:
        if is_stdout:
            print(msg, end="", file=sys.stdout)
        else:
            print(msg, end="", file=sys.stderr)


def process_wrap(cmd_str, cwd=None, handler=None):
    print(f"[ComfyUI LLaVA-NeXT Captioner] running {' '.join(cmd_str)}")
    process = subprocess.Popen(
        cmd_str,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    if handler is None:
        handler = handle_stream

    stdout_thread = threading.Thread(target=handler, args=(process.stdout, True))
    stderr_thread = threading.Thread(target=handler, args=(process.stderr, False))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.wait()


try:
    print("[ComfyUI LLaVA-NeXT Captioner] Installing dependencies")

    if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
        pip_install = [sys.executable, "-s", "-m", "pip", "install"]
    else:
        pip_install = [sys.executable, "-m", "pip", "install"]

    process_wrap(pip_install + ["transformers", "torch"])

except Exception as e:
    print(
        "[ComfyUI LLaVA-NeXT Captioner] Dependency installation has failed. Please install manually."
    )
    traceback.print_exc()
