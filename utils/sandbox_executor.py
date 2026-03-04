import base64
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Protocol

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


class DockerSandboxExecutor:
    """Implementação de execução em sandbox usando Docker local."""

    IMAGE_OUTPUT_PATTERN = r"__PYVIS_IMAGE_B64__([A-Za-z0-9+/=]+)"

    def execute(self, code: str, df: pd.DataFrame, *, timeout_seconds: int = 90) -> bytes:
        load_dotenv()
        image = (os.getenv("DOCKER_SANDBOX_IMAGE") or "pyvis-sandbox:latest").strip()
        memory = (os.getenv("DOCKER_SANDBOX_MEMORY") or "512m").strip()
        cpus = (os.getenv("DOCKER_SANDBOX_CPUS") or "1.0").strip()
        pids_limit = (os.getenv("DOCKER_SANDBOX_PIDS") or "128").strip()
        tmpfs_size = (os.getenv("DOCKER_SANDBOX_TMPFS_SIZE") or "64m").strip()

        with tempfile.TemporaryDirectory(prefix="pyvis-sandbox-") as temp_dir:
            work_dir = Path(temp_dir)
            df_path = work_dir / "df.csv"
            code_path = work_dir / "user_code.py"
            df.to_csv(df_path, index=False)
            code_path.write_text(code, encoding="utf-8")

            command = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--security-opt",
                "no-new-privileges",
                "--cap-drop",
                "ALL",
                "--pids-limit",
                pids_limit,
                "--memory",
                memory,
                "--cpus",
                cpus,
                "--tmpfs",
                f"/tmp:rw,noexec,nosuid,size={tmpfs_size}",
                "-v",
                f"{work_dir}:/work:ro",
                image,
                "python",
                "/opt/pyvis/runner.py",
                "/work/user_code.py",
                "/work/df.csv",
            ]

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Docker não está instalado ou não está no PATH. "
                    "Instale o Docker Engine e tente novamente."
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Timeout na sandbox Docker após {timeout_seconds}s."
                ) from exc

        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        if result.returncode != 0:
            lowered_error = stderr.lower()
            if "cannot connect to the docker daemon" in lowered_error:
                raise RuntimeError(
                    "Docker daemon indisponível. Inicie o serviço do Docker e tente novamente."
                )
            if "unable to find image" in lowered_error or "pull access denied" in lowered_error:
                raise RuntimeError(
                    "Imagem da sandbox Docker não encontrada. "
                    "Gere a imagem com: docker build -t pyvis-sandbox:latest docker/sandbox"
                )
            error_detail = stderr or stdout or "sem detalhes no docker run"
            raise RuntimeError(f"Erro ao executar código na sandbox Docker: {error_detail}")

        match = re.search(self.IMAGE_OUTPUT_PATTERN, stdout)
        if not match:
            raise RuntimeError(
                "A sandbox Docker executou o código, mas não retornou imagem. "
                "Verifique se o código finaliza com 'img_buffer'."
            )

        return base64.b64decode(match.group(1))


def get_sandbox_executor(kind: Literal["docker", "e2b"]) -> CodeSandboxExecutor:
    sandbox_kind = kind.lower().strip()
    if sandbox_kind == "docker":
        return DockerSandboxExecutor()
    if sandbox_kind == "e2b":
        return E2BSandboxExecutor()
    raise ValueError(f"Sandbox inválida: {kind}. Use 'docker' ou 'e2b'.")
