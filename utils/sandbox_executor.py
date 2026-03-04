import base64
import os
import re
from typing import Protocol

import pandas as pd
from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox


class CodeSandboxExecutor(Protocol):
    def execute(self, code: str, df: pd.DataFrame, *, timeout_seconds: int = 90) -> bytes:
        """Executa código Python em sandbox e retorna a imagem gerada em bytes."""


class E2BSandboxExecutor:
    """Implementação de execução em sandbox usando E2B."""

    IMAGE_OUTPUT_PATTERN = r"__PYVIS_IMAGE_B64__([A-Za-z0-9+/=]+)"

    def execute(self, code: str, df: pd.DataFrame, *, timeout_seconds: int = 90) -> bytes:
        load_dotenv()
        if not (os.getenv("E2B_API_KEY") or "").strip():
            raise ValueError("E2B_API_KEY não configurada.")

        wrapped_code = self._build_wrapped_code(code)

        sandbox = Sandbox.create()
        try:
            sandbox.files.write("/tmp/df.csv", df.to_csv(index=False))
            execution = sandbox.run_code(wrapped_code, timeout=timeout_seconds)
        finally:
            sandbox.kill()

        if execution.error:
            raise RuntimeError(f"Erro ao executar código na sandbox E2B: {execution.error}")

        stdout = execution.logs.stdout if execution.logs else []
        output_text = "\n".join(str(line) for line in stdout)
        match = re.search(self.IMAGE_OUTPUT_PATTERN, output_text)
        if not match:
            raise RuntimeError(
                "A sandbox executou o código, mas não retornou imagem. "
                "Verifique se o código finaliza com 'img_buffer'."
            )

        return base64.b64decode(match.group(1))

    @staticmethod
    def _build_wrapped_code(code: str) -> str:
        return (
            "import pandas as pd\n"
            "df = pd.read_csv('/tmp/df.csv')\n\n"
            f"{code}\n\n"
            "if 'img_buffer' not in locals():\n"
            "    raise RuntimeError(\"O código não gerou a variável 'img_buffer'.\")\n"
            "if hasattr(img_buffer, 'getvalue'):\n"
            "    _img_bytes = img_buffer.getvalue()\n"
            "elif isinstance(img_buffer, (bytes, bytearray)):\n"
            "    _img_bytes = bytes(img_buffer)\n"
            "else:\n"
            "    raise RuntimeError(\"'img_buffer' deve ser BytesIO ou bytes.\")\n"
            "import base64 as _base64\n"
            "print('__PYVIS_IMAGE_B64__' + _base64.b64encode(_img_bytes).decode('ascii'))\n"
        )
