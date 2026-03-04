import os
import base64
import streamlit as st
import pandas as pd
import plotly.io as pio
import google.generativeai as genai
from dotenv import load_dotenv
from utils import (
    CodeSandboxExecutor,
    E2BSandboxExecutor,
    MatplotlibVisualizationPromptBuilder,
    PlotlyVisualizationPromptBuilder,
    VisualizationPromptBuilder,
)


# ------------------------------------------------------------------
# Helpers de compatibilidade e estilo
# ------------------------------------------------------------------

def _rerun():
    """Streamlit 1.32+ renomeou experimental_rerun -> rerun."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover
        st.experimental_rerun()


def inject_styles():
    st.markdown(
        """
        <style>
        .stCodeBlock {max-height: 240px; overflow-y: auto;}
        .chat-img {max-width: 420px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


CHAT_IMAGE_WIDTH = 420  # só controla exibição; download mantém resolução
sandbox_executor: CodeSandboxExecutor = E2BSandboxExecutor()
prompt_builders: dict[str, VisualizationPromptBuilder] = {
    "Imagem (Matplotlib)": MatplotlibVisualizationPromptBuilder(),
    "Interativa (Plotly)": PlotlyVisualizationPromptBuilder(),
}


# ------------------------------------------------------------------
# Estado e configuração
# ------------------------------------------------------------------

def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []  # legado
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("dataset_name", None)
    st.session_state.setdefault("api_key", os.getenv("GOOGLE_API_KEY", ""))
    st.session_state.setdefault("favorites", [])
    st.session_state.setdefault("page", "Dados")
    st.session_state.setdefault("global_prompt", "")
    st.session_state.setdefault("viz_mode", "Imagem (Matplotlib)")


def get_model():
    api_key = st.session_state.get("api_key")
    if not api_key:
        st.warning("Defina a chave de API na página Configurações.")
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-pro")


# ------------------------------------------------------------------
# Favoritos
# ------------------------------------------------------------------

def add_favorite(
    img_bytes: bytes,
    code: str | None,
    title: str,
    *,
    plotly_json: str | None = None,
    viz_mode: str = "Imagem (Matplotlib)",
):
    fav = {
        "title": title or "Visualização",
        "image": base64.b64encode(img_bytes).decode(),
        "code": code,
        "plotly_json": plotly_json,
        "viz_mode": viz_mode,
    }
    if fav not in st.session_state.favorites:
        st.session_state.favorites.append(fav)
        return True
    return False


def favorite_callback(
    img_bytes_or_b64,
    code,
    title,
    feedback_key: str,
    plotly_json: str | None = None,
    viz_mode: str = "Imagem (Matplotlib)",
):
    img_bytes = (
        base64.b64decode(img_bytes_or_b64)
        if isinstance(img_bytes_or_b64, str)
        else img_bytes_or_b64
    )
    added = add_favorite(
        img_bytes,
        code,
        title,
        plotly_json=plotly_json,
        viz_mode=viz_mode,
    )
    st.session_state[feedback_key] = "added" if added else "exists"


# ------------------------------------------------------------------
# Utilidades de visualização/código
# ------------------------------------------------------------------

def build_prompt(df: pd.DataFrame, user_prompt: str, viz_mode: str) -> str:
    prompt_builder = prompt_builders[viz_mode]
    return prompt_builder.build_prompt(
        df,
        user_prompt,
        global_instructions=st.session_state.get("global_prompt", ""),
    )


def clean_code(raw: str) -> str:
    code = raw.strip()
    if code.startswith("```python"):
        code = code.split("```python", 1)[1]
    if code.startswith("```"):
        code = code.split("```", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    return code.strip()


def execute_visualization(code: str, df: pd.DataFrame, viz_mode: str):
    if viz_mode == "Interativa (Plotly)":
        plotly_json, plotly_image = sandbox_executor.execute_plotly(
            code,
            df,
            timeout_seconds=90,
        )
        return {"plotly_json": plotly_json, "image": plotly_image}

    image_bytes = sandbox_executor.execute(
        code,
        df,
        timeout_seconds=90,
    )
    return {"image": image_bytes}


def show_dataset_preview(df: pd.DataFrame, name: str | None):
    st.info(f"📌 Usando {name or 'dataset carregado'} ({len(df)} linhas, {len(df.columns)} colunas).")
    st.markdown("**Amostra (10 linhas):**")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("**Colunas disponíveis:**")
    col_info = pd.DataFrame(
        {"Coluna": df.columns, "Tipo": df.dtypes.astype(str), "Não-nulos": df.count().values}
    )
    st.dataframe(col_info, use_container_width=True)


# ------------------------------------------------------------------
# Layout e navegação
# ------------------------------------------------------------------

def render_sidebar():
    st.sidebar.header("Navegação")

    def set_page(name: str):
        st.session_state.page = name

    btn_style = {"use_container_width": True}
    st.sidebar.button(
        "📁 Dados",
        type="primary" if st.session_state.page == "Dados" else "secondary",
        on_click=set_page,
        args=("Dados",),
        **btn_style,
    )
    st.sidebar.button(
        "💬 Chat de Visualização",
        type="primary" if st.session_state.page == "Chat de Visualização" else "secondary",
        on_click=set_page,
        args=("Chat de Visualização",),
        **btn_style,
    )
    st.sidebar.button(
        "⭐ Favoritos",
        type="primary" if st.session_state.page == "Favoritos" else "secondary",
        on_click=set_page,
        args=("Favoritos",),
        **btn_style,
    )
    st.sidebar.button(
        "⚙️ Configurações",
        type="primary" if st.session_state.page == "Configurações" else "secondary",
        on_click=set_page,
        args=("Configurações",),
        **btn_style,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Como usar")
    st.sidebar.markdown(
        "1. Vá em **Dados** e carregue um CSV.\n"
        "2. Abra **Chat de Visualização** para pedir gráficos.\n"
        "3. Em **Configurações**, defina a chave de API (opcional, se já estiver no .env)."
    )


# ------------------------------------------------------------------
# Páginas
# ------------------------------------------------------------------

def page_dados():
    st.subheader("📁 Upload e Visualização do CSV")
    uploaded_file = st.file_uploader("Carregue seu arquivo CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.dataset_name = uploaded_file.name
            st.success(f"✅ Arquivo carregado: {uploaded_file.name} ({len(df)} linhas, {len(df.columns)} colunas)")
            show_dataset_preview(df, uploaded_file.name)
        except Exception as e:
            st.error(f"❌ Erro ao carregar arquivo: {str(e)}")
    elif st.session_state.df is not None:
        show_dataset_preview(st.session_state.df, st.session_state.dataset_name)
    else:
        st.info("Envie um CSV para habilitar o chat de visualização.")


def render_message(message: dict, idx: int):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "plotly_json" in message:
            figure = pio.from_json(message["plotly_json"])
            st.plotly_chart(figure, use_container_width=True)
        if message["role"] == "assistant" and "image" in message:
            img_bytes = base64.b64decode(message["image"])
            st.image(img_bytes, width=CHAT_IMAGE_WIDTH)
            st.download_button(
                label="⬇️ Baixar visualização",
                data=img_bytes,
                file_name=f"visualizacao_{idx}.png",
                mime="image/png",
                key=f"download_{idx}",
            )
            feedback_key = f"fav_feedback_{idx}"
            st.button(
                "⭐ Favoritar",
                key=f"fav_{idx}",
                on_click=favorite_callback,
                args=(
                    message["image"],
                    message.get("code"),
                    message.get("title"),
                    feedback_key,
                    message.get("plotly_json"),
                    message.get("viz_mode", "Imagem (Matplotlib)"),
                ),
            )
            if st.session_state.get(feedback_key) == "added":
                st.success("Adicionada aos favoritos!")
            elif st.session_state.get(feedback_key) == "exists":
                st.info("Já estava nos favoritos.")
        if "code" in message:
            with st.expander("Ver código"):
                st.code(message["code"], language="python")


def page_chat():
    st.subheader("💬 Chat de Visualizações")
    st.selectbox(
        "Tipo de visualização",
        options=list(prompt_builders.keys()),
        key="viz_mode",
    )

    # Histórico
    for idx, message in enumerate(st.session_state.messages):
        render_message(message, idx)

    user_prompt = st.chat_input("Descreva a visualização desejada")
    if not user_prompt:
        return

    if st.session_state.df is None:
        st.warning("Envie um CSV na página Dados antes de pedir uma visualização.")
        return

    model = get_model()
    if model is None:
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Gerando visualização com Gemini..."):
            try:
                df = st.session_state.df
                viz_mode = st.session_state.viz_mode
                prompt = build_prompt(df, user_prompt, viz_mode)
                response = model.generate_content(prompt)
                code = clean_code(response.text)

                result = execute_visualization(code, df, viz_mode)
                st.write("Aqui está sua visualização:")
                assistant_payload = {
                    "role": "assistant",
                    "content": "Visualização gerada!",
                    "title": user_prompt,
                    "code": code,
                    "viz_mode": viz_mode,
                }
                if "plotly_json" in result:
                    figure = pio.from_json(result["plotly_json"])
                    st.plotly_chart(figure, use_container_width=True)
                    assistant_payload["plotly_json"] = result["plotly_json"]

                namespace_img = result["image"]
                st.image(namespace_img, width=CHAT_IMAGE_WIDTH)
                st.download_button(
                    label="⬇️ Baixar visualização",
                    data=namespace_img,
                    file_name="visualizacao.png",
                    mime="image/png",
                )

                feedback_key = "fav_feedback_new"
                st.button(
                    "⭐ Favoritar",
                    key=f"fav_new_{viz_mode}",
                    on_click=favorite_callback,
                    args=(
                        namespace_img,
                        code,
                        user_prompt,
                        feedback_key,
                        result.get("plotly_json"),
                        viz_mode,
                    ),
                )
                if st.session_state.get(feedback_key) == "added":
                    st.success("Adicionada aos favoritos!")
                elif st.session_state.get(feedback_key) == "exists":
                    st.info("Já estava nos favoritos.")
                assistant_payload["image"] = base64.b64encode(namespace_img).decode()

                with st.expander("Ver código"):
                    st.code(code, language="python")

                st.session_state.messages.append(assistant_payload)
            except Exception as e:
                st.error(f"❌ Erro ao gerar ou executar o código: {str(e)}")


def page_config():
    st.subheader("⚙️ Configurações da Aplicação")
    st.write("Informe sua chave de API do Google Gemini (válida apenas para esta sessão).")
    new_key = st.text_input("GOOGLE_API_KEY", type="password", value=st.session_state.api_key or "")
    if st.button("Salvar chave de API"):
        st.session_state.api_key = new_key.strip()
        if st.session_state.api_key:
            st.success("Chave de API salva para esta sessão.")
        else:
            st.warning("Chave vazia. Configure uma chave válida para usar o chat.")

    st.write("Prompt global (será enviado antes de cada solicitação ao LLM).")
    global_prompt = st.text_area(
        "Prompt global",
        value=st.session_state.global_prompt or "",
        placeholder="Ex.: Sempre use títulos claros e cores acessíveis...",
        height=120,
    )
    if st.button("Salvar prompt global"):
        st.session_state.global_prompt = global_prompt.strip()
        st.success("Prompt global salvo para esta sessão.")

    st.info("Opcional: defina GOOGLE_API_KEY no arquivo .env antes de iniciar o app.")


def page_favoritos():
    st.subheader("⭐ Visualizações Favoritas")
    if not st.session_state.favorites:
        st.info("Nenhuma visualização favoritada ainda. Gere um gráfico e clique em ⭐ Favoritar.")
        return

    cols = st.columns(3)
    for i, fav in enumerate(st.session_state.favorites):
        col = cols[i % 3]
        with col:
            if fav.get("plotly_json"):
                figure = pio.from_json(fav["plotly_json"])
                st.plotly_chart(figure, use_container_width=True)
            img_bytes = base64.b64decode(fav["image"])
            st.image(img_bytes, use_container_width=True)
            st.caption(f"{fav.get('title') or f'Favorito {i+1}'} ({fav.get('viz_mode', 'Imagem (Matplotlib)')})")
            st.download_button(
                label="⬇️ Baixar",
                data=img_bytes,
                file_name=f"favorito_{i+1}.png",
                mime="image/png",
                key=f"fav_dl_{i}",
            )
            if st.button("🗑️ Remover", key=f"fav_rm_{i}"):
                st.session_state.favorites.pop(i)
                _rerun()
            with st.expander("Ver código"):
                st.code(fav.get("code") or "Código não disponível", language="python")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    load_dotenv()
    st.set_page_config(page_title="Gerador de Visualizações com IA", layout="wide")
    st.title("🎨 Gerador de Visualizações de Dados com IA")
    st.markdown("Use o menu lateral para navegar entre Dados, Chat e Configurações.")

    inject_styles()
    init_state()
    render_sidebar()

    if st.session_state.page == "Dados":
        page_dados()
    elif st.session_state.page == "Chat de Visualização":
        page_chat()
    elif st.session_state.page == "Configurações":
        page_config()
    else:
        page_favoritos()


if __name__ == "__main__":
    main()
