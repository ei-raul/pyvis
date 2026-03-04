import base64
import io
import sys
import traceback

import pandas as pd

IMAGE_OUTPUT_PREFIX = "__PYVIS_IMAGE_B64__"


def _load_code(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def _extract_image_bytes(namespace: dict) -> bytes:
    if "img_buffer" not in namespace:
        raise RuntimeError("O código não gerou a variável 'img_buffer'.")

    img_buffer = namespace["img_buffer"]
    if hasattr(img_buffer, "getvalue"):
        return img_buffer.getvalue()
    if isinstance(img_buffer, (bytes, bytearray)):
        return bytes(img_buffer)
    if isinstance(img_buffer, io.BytesIO):
        return img_buffer.getvalue()

    raise RuntimeError("'img_buffer' deve ser BytesIO ou bytes.")


def main() -> int:
    if len(sys.argv) != 3:
        print("Uso: runner.py <code_path> <df_path>", file=sys.stderr)
        return 2

    code_path = sys.argv[1]
    df_path = sys.argv[2]

    try:
        code = _load_code(code_path)
        df = pd.read_csv(df_path)
        namespace = {"df": df, "__name__": "__main__"}
        exec(compile(code, code_path, "exec"), namespace, namespace)
        img_bytes = _extract_image_bytes(namespace)
        print(IMAGE_OUTPUT_PREFIX + base64.b64encode(img_bytes).decode("ascii"))
        return 0
    except Exception:
        traceback.print_exc(limit=6, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
