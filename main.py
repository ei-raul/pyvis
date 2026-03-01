import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import json
import sys
from io import StringIO
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Carregar variáveis de ambiente
load_dotenv()

st.set_page_config(page_title="Gerador de Visualizações com IA", layout="wide")

st.title("🎨 Gerador de Visualizações de Dados com IA")
st.markdown("Carregue um CSV e descreva a visualização que deseja - a IA criará para você!")

# Carregar API key do arquivo .env
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("⚠️ Chave API do Google Gemini não encontrada! Crie um arquivo .env com GOOGLE_API_KEY=sua_chave")
    st.stop()

# Configurar o Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-pro')

# Sidebar para informações
with st.sidebar:
    st.header("⚙️ Configurações")
    st.success("✅ API Key carregada do arquivo .env")
    st.markdown("---")
    st.markdown("### 💡 Como usar")
    st.markdown("""
    1. Certifique-se de ter o arquivo .env com GOOGLE_API_KEY
    2. Faça upload de um arquivo CSV
    3. Descreva a visualização desejada
    4. Clique em 'Gerar Visualização'
    """)
    st.markdown("---")
    st.markdown("### 📄 Arquivo .env")
    st.code("GOOGLE_API_KEY=sua_chave_aqui", language="bash")

# Inicializar session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'df' not in st.session_state:
    st.session_state.df = None

# Upload do CSV
uploaded_file = st.file_uploader("📁 Carregue seu arquivo CSV", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        st.success(f"✅ Arquivo carregado: {uploaded_file.name} ({len(df)} linhas, {len(df.columns)} colunas)")
        
        # Preview dos dados
        with st.expander("👀 Visualizar dados"):
            st.dataframe(df.head(10))
            st.markdown("**Colunas disponíveis:**")
            col_info = pd.DataFrame({
                'Coluna': df.columns,
                'Tipo': df.dtypes.astype(str),
                'Não-nulos': df.count().values
            })
            st.dataframe(col_info)
            
    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivo: {str(e)}")

# Campo de descrição da visualização
if st.session_state.df is not None:
    st.markdown("---")
    st.subheader("📝 Descreva a visualização desejada")
    
    user_prompt = st.text_area(
        "Descrição",
        placeholder="Exemplo: Crie um gráfico de barras mostrando as vendas por categoria, com cores diferentes para cada categoria",
        height=100
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        generate_btn = st.button("🎨 Gerar Visualização", type="primary", use_container_width=True)
    
    if generate_btn:
        if not user_prompt:
            st.error("⚠️ Por favor, descreva a visualização desejada")
        else:
            with st.spinner("🤖 A IA está gerando o código da visualização..."):
                try:
                    # Preparar informações do dataset
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
                    
                    # Criar prompt para o Gemini
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
                    
                    # Chamar API do Google Gemini
                    response = model.generate_content(prompt)
                    code = response.text.strip()
                        
                    # Limpar o código de possíveis marcadores markdown
                    if code.startswith('```python'):
                        code = code.split('```python')[1]
                    if code.startswith('```'):
                        code = code.split('```')[1]
                    if code.endswith('```'):
                        code = code.rsplit('```', 1)[0]
                    code = code.strip()
                    
                    st.markdown("### 💻 Código gerado:")
                    st.code(code, language='python')
                    
                    # Executar o código em um ambiente controlado
                    st.markdown("### 🎨 Visualização gerada:")
                    
                    try:
                        # Preparar namespace para execução
                        namespace = {
                            'df': df,
                            'pd': pd,
                            'plt': plt,
                            'io': io,
                            'BytesIO': io.BytesIO
                        }
                        
                        # Executar código
                        exec(code, namespace)
                        
                        # Obter imagem
                        if 'img_buffer' in namespace:
                            img_buffer = namespace['img_buffer']
                            st.image(img_buffer, use_container_width=True)
                            
                            # Adicionar ao histórico
                            st.session_state.history.append({
                                'prompt': user_prompt,
                                'code': code,
                                'image': base64.b64encode(img_buffer.getvalue()).decode()
                            })
                            
                            # Botão de download
                            st.download_button(
                                label="⬇️ Baixar visualização",
                                data=img_buffer.getvalue(),
                                file_name="visualizacao.png",
                                mime="image/png"
                            )
                            
                            st.success("✅ Visualização gerada com sucesso!")
                        else:
                            st.error("❌ O código não gerou a variável 'img_buffer' esperada")
                            
                    except Exception as e:
                        st.error(f"❌ Erro ao executar o código gerado: {str(e)}")
                        st.code(str(e))
                    
                except Exception as e:
                    st.error(f"❌ Erro ao gerar código: {str(e)}")

# Histórico de visualizações
if st.session_state.history:
    st.markdown("---")
    st.subheader("📚 Histórico de Visualizações")
    
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Visualização {len(st.session_state.history) - i}: {item['prompt'][:50]}..."):
            st.markdown(f"**Prompt:** {item['prompt']}")
            st.code(item['code'], language='python')
            img_data = base64.b64decode(item['image'])
            st.image(img_data, use_container_width=True)
            st.download_button(
                label="⬇️ Baixar",
                data=img_data,
                file_name=f"visualizacao_{i}.png",
                mime="image/png",
                key=f"download_{i}"
            )