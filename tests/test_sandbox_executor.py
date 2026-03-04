import base64
import subprocess
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from utils.sandbox_executor import (
    DockerSandboxExecutor,
    E2BSandboxExecutor,
    get_sandbox_executor,
)


class DockerSandboxExecutorTests(unittest.TestCase):
    def setUp(self):
        self.executor = DockerSandboxExecutor()
        self.df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        self.code = "img_buffer = b'abc'"

    @patch("utils.sandbox_executor.subprocess.run")
    def test_execute_returns_png_bytes(self, mock_run):
        payload = base64.b64encode(b"png-bytes").decode("ascii")
        mock_run.return_value = Mock(returncode=0, stdout=f"__PYVIS_IMAGE_B64__{payload}\n", stderr="")

        image_bytes = self.executor.execute(self.code, self.df)

        self.assertEqual(image_bytes, b"png-bytes")
        self.assertTrue(mock_run.called)

    @patch("utils.sandbox_executor.subprocess.run")
    def test_execute_raises_when_output_marker_is_missing(self, mock_run):
        mock_run.return_value = Mock(returncode=0, stdout="sem marcador", stderr="")

        with self.assertRaises(RuntimeError) as ctx:
            self.executor.execute(self.code, self.df)

        self.assertIn("não retornou imagem", str(ctx.exception))

    @patch("utils.sandbox_executor.subprocess.run")
    def test_execute_raises_on_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["docker"], timeout=90)

        with self.assertRaises(RuntimeError) as ctx:
            self.executor.execute(self.code, self.df, timeout_seconds=90)

        self.assertIn("Timeout", str(ctx.exception))

    @patch("utils.sandbox_executor.subprocess.run")
    def test_execute_raises_when_docker_is_missing(self, mock_run):
        mock_run.side_effect = FileNotFoundError("docker")

        with self.assertRaises(RuntimeError) as ctx:
            self.executor.execute(self.code, self.df)

        self.assertIn("Docker não está instalado", str(ctx.exception))

    @patch("utils.sandbox_executor.subprocess.run")
    def test_execute_raises_when_daemon_is_unavailable(self, mock_run):
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
        )

        with self.assertRaises(RuntimeError) as ctx:
            self.executor.execute(self.code, self.df)

        self.assertIn("daemon", str(ctx.exception).lower())


class SandboxFactoryTests(unittest.TestCase):
    def test_factory_returns_docker_executor(self):
        executor = get_sandbox_executor("docker")
        self.assertIsInstance(executor, DockerSandboxExecutor)

    def test_factory_returns_e2b_executor(self):
        executor = get_sandbox_executor("e2b")
        self.assertIsInstance(executor, E2BSandboxExecutor)

    def test_factory_raises_for_invalid_kind(self):
        with self.assertRaises(ValueError):
            get_sandbox_executor("invalid")


if __name__ == "__main__":
    unittest.main()
