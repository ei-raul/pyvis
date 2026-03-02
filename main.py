import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Compat: rerun helper (Streamlit renamed experimental_rerun -> rerun)
def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover
        st.experimental_rerun()

# Carregar variáveis de ambiente
load_dotenv()

st.set_page_config(page_title="Gerador de Visualizações com IA", layout="wide")

st.title("🎨 Gerador de Visualizações de Dados com IA")
st.markdown("Use o menu lateral para navegar entre Dados, Chat e Configurações.")

# Estilos globais: limitar altura de blocos de código e reduzir exibição das imagens no chat
st.markdown(
    """
    <style>
    .stCodeBlock {max-height: 240px; overflow-y: auto;}
    .chat-img {max-width: 420px;}
    </style>
    """,
    unsafe_allow_html=True,
)

CHAT_IMAGE_WIDTH = 420  # largura apenas para exibição no chat (resolução original preservada)

# Inicializar session state
if 'history' not in st.session_state:
    st.session_state.history = []  # Histórico legado (mantido para compatibilidade)
if 'messages' not in st.session_state:
    st.session_state.messages = []  # Histórico no formato de chat
if 'df' not in st.session_state:
    st.session_state.df = None
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("GOOGLE_API_KEY", "")
if 'favorites' not in st.session_state:
    st.session_state.favorites = []


def get_model():
    """Instancia o modelo Gemini com a chave disponível na sessão."""
    api_key = st.session_state.get("api_key")
    if not api_key:
        st.warning("Defina a chave de API na página Configurações.")
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-pro')


# Sidebar
st.sidebar.header("Navegação")
if "page" not in st.session_state:
    st.session_state.page = "Dados"

def set_page(name: str):
    st.session_state.page = name


def add_favorite(img_bytes: bytes, code: str | None, title: str):
    """Adiciona visualização aos favoritos evitando duplicados exatos."""
    fav = {
        'title': title or "Visualização",
        'image': base64.b64encode(img_bytes).decode(),
        'code': code,
    }
    if fav not in st.session_state.favorites:
        st.session_state.favorites.append(fav)
        return True
    return False


def favorite_callback(img_bytes_or_b64, code, title, feedback_key: str):
    """Callback para botões de favoritar com feedback persistente."""
    img_bytes = (
        base64.b64decode(img_bytes_or_b64)
        if isinstance(img_bytes_or_b64, str)
        else img_bytes_or_b64
    )
    added = add_favorite(img_bytes, code, title)
    st.session_state[feedback_key] = "added" if added else "exists"

btn_style = {"use_container_width": True}
st.sidebar.button("📁 Dados", type="primary" if st.session_state.page == "Dados" else "secondary",
                  on_click=set_page, args=("Dados",), **btn_style)
st.sidebar.button("💬 Chat de Visualização", type="primary" if st.session_state.page == "Chat de Visualização" else "secondary",
                  on_click=set_page, args=("Chat de Visualização",), **btn_style)
st.sidebar.button("⭐ Favoritos", type="primary" if st.session_state.page == "Favoritos" else "secondary",
                  on_click=set_page, args=("Favoritos",), **btn_style)
st.sidebar.button("⚙️ Configurações", type="primary" if st.session_state.page == "Configurações" else "secondary",
                  on_click=set_page, args=("Configurações",), **btn_style)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Como usar")
st.sidebar.markdown(
    "1. Vá em **Dados** e carregue um CSV.\n"
    "2. Abra **Chat de Visualização** para pedir gráficos.\n"
    "3. Em **Configurações**, defina a chave de API (opcional, se já estiver no .env)."
)

# Página: Dados
if st.session_state.page == "Dados":
    st.subheader("📁 Upload e Visualização do CSV")
    st.markdown('<div class="pane dataset-pane">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Carregue seu arquivo CSV", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.dataset_name = uploaded_file.name

            st.success(f"✅ Arquivo carregado: {uploaded_file.name} ({len(df)} linhas, {len(df.columns)} colunas)")

            st.markdown("**Amostra (10 linhas):**")
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown("**Colunas disponíveis:**")
            col_info = pd.DataFrame({
                'Coluna': df.columns,
                'Tipo': df.dtypes.astype(str),
                'Não-nulos': df.count().values
            })
            st.dataframe(col_info, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Erro ao carregar arquivo: {str(e)}")
    elif st.session_state.df is not None:
        df = st.session_state.df
        name = st.session_state.dataset_name or "dataset carregado"
        st.info(f"📌 Usando {name} já carregado ({len(df)} linhas, {len(df.columns)} colunas).")
        st.markdown("**Amostra (10 linhas):**")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("**Colunas disponíveis:**")
        col_info = pd.DataFrame({
            'Coluna': df.columns,
            'Tipo': df.dtypes.astype(str),
            'Não-nulos': df.count().values
        })
        st.dataframe(col_info, use_container_width=True)
    else:
        st.info("Envie um CSV para habilitar o chat de visualização.")

    st.markdown('</div>', unsafe_allow_html=True)

# Página: Chat
elif st.session_state.page == "Chat de Visualização":
    st.subheader("💬 Chat de Visualizações")
    st.markdown('<div class="pane chat-pane">', unsafe_allow_html=True)

    # Histórico
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message['role']):
            if message['role'] == 'user':
                st.write(message['content'])
            else:
                st.write(message['content'])
                if 'image' in message:
                    img_bytes = base64.b64decode(message['image'])
                    st.image(img_bytes, width=CHAT_IMAGE_WIDTH)
                    st.download_button(
                        label="⬇️ Baixar visualização",
                        data=img_bytes,
                        file_name=f"visualizacao_{idx}.png",
                        mime="image/png",
                        key=f"download_{idx}"
                    )
                    feedback_key = f"fav_feedback_{idx}"
                    st.button(
                        "⭐ Favoritar",
                        key=f"fav_{idx}",
                        on_click=favorite_callback,
                        args=(message['image'], message.get('code'), message.get('title'), feedback_key),
                    )
                    if st.session_state.get(feedback_key) == "added":
                        st.success("Adicionada aos favoritos!")
                    elif st.session_state.get(feedback_key) == "exists":
                        st.info("Já estava nos favoritos.")
                if 'code' in message:
                    with st.expander("Ver código"):
                        st.code(message['code'], language='python')

    st.markdown('</div>', unsafe_allow_html=True)

    user_prompt = st.chat_input("Descreva a visualização desejada")

    if user_prompt:
        if st.session_state.df is None:
            st.warning("Envie um CSV na página Dados antes de pedir uma visualização.")
        else:
            model = get_model()
            if model is None:
                st.stop()

            st.session_state.messages.append({'role': 'user', 'content': user_prompt})
            with st.chat_message('user'):
                st.write(user_prompt)

            with st.chat_message('assistant'):
                with st.spinner("Gerando visualização com Gemini..."):
                    try:
                        df = st.session_state.df
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

                        prompt = f"""
                            Você é um especialista em visualização de dados com Python e Matplotlib.

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

                        response = model.generate_content(prompt)
                        code = response.text.strip()

                        if code.startswith('```python'):
                            code = code.split('```python')[1]
                        if code.startswith('```'):
                            code = code.split('```')[1]
                        if code.endswith('```'):
                            code = code.rsplit('```', 1)[0]
                        code = code.strip()

                        st.write("Aqui está sua visualização:")

                        namespace = {'df': df, 'pd': pd, 'plt': plt, 'io': io, 'BytesIO': io.BytesIO}
                        exec(code, namespace)

                        if 'img_buffer' in namespace:
                            img_buffer = namespace['img_buffer']
                            img_bytes = img_buffer.getvalue()
                            st.image(img_bytes, width=CHAT_IMAGE_WIDTH)
                            st.download_button(
                                label="⬇️ Baixar visualização",
                                data=img_bytes,
                                file_name="visualizacao.png",
                                mime="image/png"
                            )
                            feedback_key = "fav_feedback_new"
                            st.button(
                                "⭐ Favoritar",
                                key="fav_new",
                                on_click=favorite_callback,
                                args=(img_bytes, code, user_prompt, feedback_key),
                            )
                            if st.session_state.get(feedback_key) == "added":
                                st.success("Adicionada aos favoritos!")
                            elif st.session_state.get(feedback_key) == "exists":
                                st.info("Já estava nos favoritos.")
                            with st.expander("Ver código"):
                                st.code(code, language='python')

                            st.session_state.messages.append({
                                'role': 'assistant',
                                'content': "Visualização gerada!",
                                'title': user_prompt,
                                'code': code,
                                'image': base64.b64encode(img_bytes).decode()
                            })
                        else:
                            st.error("❌ O código não gerou a variável 'img_buffer' esperada")
                    except Exception as e:
                        st.error(f"❌ Erro ao gerar ou executar o código: {str(e)}")

# Página: Configurações
elif st.session_state.page == "Configurações":
    st.subheader("⚙️ Configurações da Aplicação")
    st.markdown('<div class="pane chat-pane">', unsafe_allow_html=True)

    st.write("Informe sua chave de API do Google Gemini (válida apenas para esta sessão).")
    new_key = st.text_input("GOOGLE_API_KEY", type="password", value=st.session_state.api_key or "")

    if st.button("Salvar chave de API"):
        st.session_state.api_key = new_key.strip()
        if st.session_state.api_key:
            st.success("Chave de API salva para esta sessão.")
        else:
            st.warning("Chave vazia. Configure uma chave válida para usar o chat.")

    st.info("Opcional: defina GOOGLE_API_KEY no arquivo .env antes de iniciar o app.")
    st.markdown('</div>', unsafe_allow_html=True)

# Página: Favoritos
else:
    st.subheader("⭐ Visualizações Favoritas")
    if not st.session_state.favorites:
        st.info("Nenhuma visualização favoritada ainda. Gere um gráfico e clique em ⭐ Favoritar.")
    else:
        cols = st.columns(3)
        for i, fav in enumerate(st.session_state.favorites):
            col = cols[i % 3]
            with col:
                img_bytes = base64.b64decode(fav['image'])
                st.image(img_bytes, use_container_width=True)
                st.caption(fav.get('title') or f"Favorito {i+1}")
                st.download_button(
                    label="⬇️ Baixar",
                    data=img_bytes,
                    file_name=f"favorito_{i+1}.png",
                    mime="image/png",
                    key=f"fav_dl_{i}"
                )
                if st.button("🗑️ Remover", key=f"fav_rm_{i}"):
                    st.session_state.favorites.pop(i)
                    _rerun()
                with st.expander("Ver código"):
                    st.code(fav.get('code') or "Código não disponível", language='python')
