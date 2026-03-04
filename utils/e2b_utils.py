import pandas as pd

from .sandbox_executor import E2BSandboxExecutor


def execute_code_in_e2b(
    code: str,
    df: pd.DataFrame,
    *,
    timeout_seconds: int = 90,
) -> bytes:
    """Wrapper de compatibilidade para execução via E2B."""
    return E2BSandboxExecutor().execute(code, df, timeout_seconds=timeout_seconds)
