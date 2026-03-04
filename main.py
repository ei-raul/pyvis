import os
import base64
import streamlit as st
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from utils import (
    get_sandbox_executor,
    get_python_environment_info,
    get_today_date_now,
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
    st.session_state.setdefault("sandbox_type", "docker")


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

def add_favorite(img_bytes: bytes, code: str | None, title: str):
    fav = {
        "title": title or "Visualização",
        "image": base64.b64encode(img_bytes).decode(),
        "code": code,
    }
    if fav not in st.session_state.favorites:
        st.session_state.favorites.append(fav)
        return True
    return False


def favorite_callback(img_bytes_or_b64, code, title, feedback_key: str):
    img_bytes = (
        base64.b64decode(img_bytes_or_b64)
        if isinstance(img_bytes_or_b64, str)
        else img_bytes_or_b64
    )
    added = add_favorite(img_bytes, code, title)
    st.session_state[feedback_key] = "added" if added else "exists"


# ------------------------------------------------------------------
# Utilidades de visualização/código
# ------------------------------------------------------------------

def build_prompt(df: pd.DataFrame, user_prompt: str) -> str:
    global_instructions = st.session_state.get("global_prompt", "").strip()
    global_block = (
        f"Instruções globais definidas pelo usuário: {global_instructions}\n\n"
        if global_instructions
        else ""
    )
    environment_info = get_python_environment_info()
    today_date = get_today_date_now()

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

    return f"""
        Você é um especialista em visualização de dados com Python e Matplotlib.

        {global_block}
        Data de hoje (obtida via datetime.now()): {today_date}

        Ambiente Python disponível para execução do código:
        {environment_info}

        {dataset_info}

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


def clean_code(raw: str) -> str:
    code = raw.strip()
    if code.startswith("```python"):
        code = code.split("```python", 1)[1]
    if code.startswith("```"):
        code = code.split("```", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    return code.strip()


def execute_code(code: str, df: pd.DataFrame, sandbox_type: str):
    executor = get_sandbox_executor(sandbox_type.lower().strip())
    return executor.execute(
        code,
        df,
        timeout_seconds=90,
    )


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
                args=(message["image"], message.get("code"), message.get("title"), feedback_key),
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
                prompt = build_prompt(df, user_prompt)
                response = model.generate_content(prompt)
                code = clean_code(response.text)

                namespace_img = execute_code(code, df, st.session_state.sandbox_type)
                if namespace_img is None:
                    st.error("❌ O código não gerou a variável 'img_buffer' esperada")
                    return

                st.write("Aqui está sua visualização:")
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
                    key="fav_new",
                    on_click=favorite_callback,
                    args=(namespace_img, code, user_prompt, feedback_key),
                )
                if st.session_state.get(feedback_key) == "added":
                    st.success("Adicionada aos favoritos!")
                elif st.session_state.get(feedback_key) == "exists":
                    st.info("Já estava nos favoritos.")

                with st.expander("Ver código"):
                    st.code(code, language="python")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Visualização gerada!",
                        "title": user_prompt,
                        "code": code,
                        "image": base64.b64encode(namespace_img).decode(),
                    }
                )
            except Exception as e:
                st.error(f"❌ Erro ao gerar ou executar o código: {str(e)}")


def page_config():
    st.subheader("⚙️ Configurações da Aplicação")
    sandbox_label = st.selectbox(
        "Sandbox de execução",
        options=["docker", "e2b"],
        format_func=lambda value: "Docker (local)" if value == "docker" else "E2B (remota)",
        index=0 if st.session_state.sandbox_type == "docker" else 1,
    )
    st.session_state.sandbox_type = sandbox_label
    if st.session_state.sandbox_type == "docker":
        st.caption(
            "Modo Docker requer imagem local da sandbox. "
            "Build: `docker build -t pyvis-sandbox:latest docker/sandbox`"
        )
    else:
        st.caption("Modo E2B requer `E2B_API_KEY` configurada.")

    st.markdown("---")
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
            img_bytes = base64.b64decode(fav["image"])
            st.image(img_bytes, use_container_width=True)
            st.caption(fav.get("title") or f"Favorito {i+1}")
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
