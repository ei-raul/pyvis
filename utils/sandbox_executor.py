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

    def execute_plotly(
        self,
        code: str,
        df: pd.DataFrame,
        *,
        timeout_seconds: int = 90,
    ) -> tuple[str, bytes]:
        """Executa código Python em sandbox e retorna (figura Plotly em JSON, imagem PNG)."""


class E2BSandboxExecutor:
    """Implementação de execução em sandbox usando E2B."""

    IMAGE_OUTPUT_PATTERN = r"__PYVIS_IMAGE_B64__([A-Za-z0-9+/=]+)"
    PLOTLY_JSON_OUTPUT_PATTERN = r"__PYVIS_PLOTLY_JSON_B64__([A-Za-z0-9+/=]+)"
    PLOTLY_IMAGE_OUTPUT_PATTERN = r"__PYVIS_PLOTLY_IMAGE_B64__([A-Za-z0-9+/=]+)"

    def execute(self, code: str, df: pd.DataFrame, *, timeout_seconds: int = 90) -> bytes:
        load_dotenv()
        if not (os.getenv("E2B_API_KEY") or "").strip():
            raise ValueError("E2B_API_KEY não configurada.")

        wrapped_code = self._build_wrapped_code(code)
        output_text = self._run_wrapped_code(wrapped_code, df, timeout_seconds=timeout_seconds)

        match = re.search(self.IMAGE_OUTPUT_PATTERN, output_text)
        if not match:
            raise RuntimeError(
                "A sandbox executou o código, mas não retornou imagem. "
                "Verifique se o código finaliza com 'img_buffer'."
            )

        return base64.b64decode(match.group(1))

    def execute_plotly(
        self,
        code: str,
        df: pd.DataFrame,
        *,
        timeout_seconds: int = 90,
    ) -> tuple[str, bytes]:
        load_dotenv()
        if not (os.getenv("E2B_API_KEY") or "").strip():
            raise ValueError("E2B_API_KEY não configurada.")

        wrapped_code = self._build_plotly_wrapped_code(code)
        output_text = self._run_wrapped_code(wrapped_code, df, timeout_seconds=timeout_seconds)

        json_match = re.search(self.PLOTLY_JSON_OUTPUT_PATTERN, output_text)
        if not json_match:
            raise RuntimeError(
                "A sandbox executou o código, mas não retornou figura Plotly. "
                "Verifique se o código finaliza com 'fig'."
            )
        image_match = re.search(self.PLOTLY_IMAGE_OUTPUT_PATTERN, output_text)
        if not image_match:
            raise RuntimeError(
                "A figura Plotly foi gerada, mas não foi possível exportar PNG. "
                "Verifique suporte de exportação de imagem (kaleido) na sandbox."
            )

        plotly_json = base64.b64decode(json_match.group(1)).decode("utf-8")
        plotly_image = base64.b64decode(image_match.group(1))
        return plotly_json, plotly_image

    @staticmethod
    def _run_wrapped_code(wrapped_code: str, df: pd.DataFrame, *, timeout_seconds: int) -> str:
        sandbox = Sandbox.create()
        try:
            sandbox.files.write("/tmp/df.csv", df.to_csv(index=False))
            execution = sandbox.run_code(wrapped_code, timeout=timeout_seconds)
        finally:
            sandbox.kill()

        if execution.error:
            raise RuntimeError(f"Erro ao executar código na sandbox E2B: {execution.error}")

        stdout = execution.logs.stdout if execution.logs else []
        return "\n".join(str(line) for line in stdout)

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

    @staticmethod
    def _build_plotly_wrapped_code(code: str) -> str:
        return (
            "import pandas as pd\n"
            "df = pd.read_csv('/tmp/df.csv')\n\n"
            f"{code}\n\n"
            "if 'fig' not in locals():\n"
            "    raise RuntimeError(\"O código não gerou a variável 'fig'.\")\n"
            "if not hasattr(fig, 'to_json'):\n"
            "    raise RuntimeError(\"'fig' deve ser uma figura Plotly válida.\")\n"
            "_fig_json = fig.to_json()\n"
            "import importlib.util as _importlib_util\n"
            "if _importlib_util.find_spec('kaleido') is None:\n"
            "    import subprocess as _subprocess\n"
            "    import sys as _sys\n"
            "    _install_proc = _subprocess.run(\n"
            "        [_sys.executable, '-m', 'pip', 'install', '-q', 'kaleido'],\n"
            "        capture_output=True,\n"
            "        text=True,\n"
            "    )\n"
            "    if _install_proc.returncode != 0:\n"
            "        raise RuntimeError(\n"
            "            \"Falha ao instalar 'kaleido' automaticamente na sandbox. \"\n"
            "            f\"stdout={_install_proc.stdout} stderr={_install_proc.stderr}\"\n"
            "        )\n"
            "try:\n"
            "    _fig_png_bytes = fig.to_image(format='png')\n"
            "except Exception as _e:\n"
            "    raise RuntimeError(f\"Falha ao exportar PNG da figura Plotly: {_e}\")\n"
            "import base64 as _base64\n"
            "print('__PYVIS_PLOTLY_JSON_B64__' + _base64.b64encode(_fig_json.encode('utf-8')).decode('ascii'))\n"
            "print('__PYVIS_PLOTLY_IMAGE_B64__' + _base64.b64encode(_fig_png_bytes).decode('ascii'))\n"
        )
