from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd

from .env_utils import get_python_environment_info, get_today_date_now


class VisualizationPromptBuilder(Protocol):
    def build_prompt(
        self,
        df: pd.DataFrame,
        user_prompt: str,
        *,
        global_instructions: str = "",
    ) -> str:
        """Monta o prompt para geração de código de visualização."""


@dataclass
class _BaseVisualizationPromptBuilder:
    get_environment_info: Callable[[], str] = get_python_environment_info
    get_today_date: Callable[[], str] = get_today_date_now

    def _build_common_context(self, df: pd.DataFrame, global_instructions: str) -> str:
        global_instructions = global_instructions.strip()
        global_block = (
            f"Instruções globais definidas pelo usuário: {global_instructions}\n\n"
            if global_instructions
            else ""
        )
        environment_info = self.get_environment_info()
        today_date = self.get_today_date()
        dataset_info = f"""
            Dataset carregado (variável 'df' do tipo pandas.DataFrame):
            - Colunas: {list(df.columns)}
            - Tipos de dados: {df.dtypes.to_dict()}
            - Shape: {df.shape}
            - Primeiras 10 linhas de exemplo do dataset:
            {df.head(10).to_string()}

            Estatísticas:
            {df.describe().to_string()}
        """
        return (
            f"{global_block}"
            f"Data de hoje (obtida via datetime.now()): {today_date}\n\n"
            f"Ambiente Python disponível para execução do código:\n"
            f"{environment_info}\n\n"
            f"{dataset_info}\n"
        )


@dataclass
class MatplotlibVisualizationPromptBuilder(_BaseVisualizationPromptBuilder):
    def build_prompt(
        self,
        df: pd.DataFrame,
        user_prompt: str,
        *,
        global_instructions: str = "",
    ) -> str:
        context = self._build_common_context(df, global_instructions)

        return f"""
            Você é um especialista em visualização de dados com Python e Matplotlib.

            {context}

            Solicitação do usuário: {user_prompt}

            Gere um código Python completo que:
            1. Use o DataFrame 'df' que já está carregado. Não crie outro DataFrame do zero, nem modifique a variável 'df'.
            Você pode criar outro DataFrame a partir do 'df' para realizar operações sem alterar os dados;
            2. Crie a visualização solicitada usando matplotlib;
            3. Salve a figura em um objeto BytesIO chamado 'img_buffer' em formato PNG;
            4. Use plt.tight_layout() para melhor aparência;
            5. Não use plt.show().

            Retorne APENAS o código Python, sem explicações, sem markdown, sem ```python. Apenas o código puro.
            O código deve começar com 'import matplotlib.pyplot as plt' e terminar salvando em img_buffer.

            Exemplo de estrutura:
            import matplotlib.pyplot as plt
            import io

            # Seu código de visualização aqui
            fig, ax = plt.subplots(figsize=(10, 6))
            # ... código do gráfico ...

            plt.tight_layout()
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
        """


@dataclass
class PlotlyVisualizationPromptBuilder(_BaseVisualizationPromptBuilder):
    def build_prompt(
        self,
        df: pd.DataFrame,
        user_prompt: str,
        *,
        global_instructions: str = "",
    ) -> str:
        context = self._build_common_context(df, global_instructions)
        return f"""
            Você é um especialista em visualização de dados com Python e Plotly.

            {context}

            Solicitação do usuário: {user_prompt}

            Gere um código Python completo que:
            1. Use o DataFrame 'df' que já está carregado. Não crie outro DataFrame do zero, nem modifique a variável 'df'.
            Você pode criar outro DataFrame a partir do 'df' para realizar operações sem alterar os dados;
            2. Crie a visualização solicitada usando Plotly (plotly.express e/ou plotly.graph_objects);
            3. Produza uma figura interativa final na variável 'fig';
            4. Não use fig.show();
            5. Garanta que o layout esteja legível (título claro, eixos nomeados e hover útil quando aplicável).

            Retorne APENAS o código Python, sem explicações, sem markdown, sem ```python. Apenas o código puro.
            O código deve finalizar com uma variável 'fig' válida.

            Exemplo de estrutura:
            import plotly.express as px

            # Seu código de visualização aqui
            fig = px.scatter(df, x="coluna_x", y="coluna_y", color="categoria")
            fig.update_layout(title="Título do gráfico")
        """
