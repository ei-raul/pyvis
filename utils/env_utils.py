import sys
from datetime import datetime
from functools import lru_cache
from importlib import metadata


@lru_cache(maxsize=4)
def get_python_environment_info(max_packages: int = 120) -> str:
    python_version = sys.version.split()[0]

    try:
        installed_names = sorted(
            {
                dist.metadata.get("Name")
                for dist in metadata.distributions()
                if dist.metadata.get("Name")
            }
        )
    except Exception:
        installed_names = []

    highlighted_packages = [
        "pandas",
        "matplotlib",
        "numpy",
        "seaborn",
        "plotly",
        "scipy",
        "geopandas",
        "scikit-learn",
        "statsmodels",
        "streamlit",
        "google-generativeai",
    ]

    highlighted_versions = []
    for package in highlighted_packages:
        try:
            highlighted_versions.append(f"{package}=={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            highlighted_versions.append(f"{package}=não instalado")

    if installed_names:
        visible_packages = installed_names[:max_packages]
        truncated_count = len(installed_names) - len(visible_packages)
        package_list = ", ".join(visible_packages)
        if truncated_count > 0:
            package_list += f", ... (+{truncated_count} pacotes)"
    else:
        package_list = "Não foi possível listar os pacotes instalados."

    highlighted_text = ", ".join(highlighted_versions)
    return (
        f"- Versão do Python: {python_version}\n"
        f"- Versões de bibliotecas relevantes: {highlighted_text}\n"
        f"- Pacotes instalados detectados: {package_list}\n"
        "- Regra obrigatória: use apenas bibliotecas que aparecem nesta lista de pacotes instalados."
    )


def get_today_date_now() -> str:
    return datetime.now().strftime("%Y-%m-%d")

