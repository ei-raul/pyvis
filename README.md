# PyVis – Gerador de Visualizações com IA

Aplicação Streamlit em formato de chat que cria visualizações de dados com Google Gemini a partir de um CSV enviado pelo usuário. Inclui favoritos, download das imagens e gestão de chave de API via interface.

## Requisitos

- Python 3.13+
- Docker Engine (para sandbox local Docker)
- Chave da API do Google Gemini (`GOOGLE_API_KEY`)
- Dependências listadas em `pyproject.toml`

## Configuração rápida

1) Clone o repositório e entre na pasta:
   ```bash
   git clone <repo> pyvis && cd pyvis
   ```
2) Crie e ative um ambiente virtual (exemplo com `python3 -m venv .venv`):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   (ou use `uv venv` se preferir, já existe `uv.lock`.)
3) Instale as dependências:
   ```bash
   pip install -r <(python - <<'PY'
   import tomllib, json, sys; data=tomllib.load(open("pyproject.toml","rb"))
   deps=data["project"]["dependencies"]; print("\n".join(deps))
   PY
   )
   ```
   (Alternativa mais simples: `pip install .` ou `uv pip sync uv.lock`.)
4) Configure a chave da API:
   - Crie um `.env` com `GOOGLE_API_KEY=suachave`
   - ou informe a chave na página **Configurações** dentro do app.
5) (Opcional, recomendado) Prepare a imagem da sandbox Docker:
   ```bash
   docker build -t pyvis-sandbox:latest docker/sandbox
   ```

## Execução

```bash
streamlit run main.py
```
O Streamlit abrirá em `http://localhost:8501`.

## Fluxo de uso

1. **Dados**  
   - Faça upload de um CSV.  
   - Veja amostra (10 linhas) e metadados das colunas.  
   - O dataset fica em memória enquanto a sessão durar.

2. **Chat de Visualização**  
   - Descreva a visualização desejada.  
   - A IA gera o código Python (Matplotlib) e a imagem.  
   - Baixe a figura ou marque ⭐ Favoritar (imagem + código).  
   - O código aparece dentro de um acordeon “Ver código” para economizar espaço.

3. **Favoritos**  
   - Galeria em grid com todas as visualizações favoritadas.  
   - Baixe novamente ou remova itens.  
   - Cada favorito traz um “Ver código” com o script gerado.

4. **Configurações**  
   - Escolha a sandbox de execução: `Docker (local)` ou `E2B (remota)`.  
   - Defina/atualize a `GOOGLE_API_KEY` apenas para a sessão atual (útil se não quiser escrever no `.env`).

## Notas de implementação

- Modelo usado: `gemini-2.5-pro` via `google-generativeai`.
- Sandbox padrão por sessão: `Docker`.
- O modo Docker executa com restrições de segurança (`--network none`, `--read-only`, limites de CPU/memória/PIDs, sem capabilities extras).
- As imagens são exibidas menores no chat (largura ~420px) para dar espaço à conversa; o download mantém a resolução completa.
- Favoritos evitam duplicatas exatas (imagem + título + código).
- O app salva estado em `st.session_state` (dataset, mensagens, favoritos, chave de API, sandbox), preservando enquanto a sessão estiver ativa.

## Possíveis ajustes

- Trocar o modelo Gemini ou parâmetros de geração alterando `get_model()` em `main.py`.
- Ajustar largura de exibição das imagens via constante `CHAT_IMAGE_WIDTH`.
- Incluir autenticação se for expor o app publicamente.

## Solução de problemas

- **“Chave API não encontrada”**: crie `.env` com `GOOGLE_API_KEY` ou informe na página **Configurações**.  
- **“Imagem Docker não encontrada”**: rode `docker build -t pyvis-sandbox:latest docker/sandbox`.  
- **“Docker daemon indisponível”**: inicie o serviço do Docker (ex.: Docker Desktop ou `systemctl start docker`).  
- **Erro de execução do código gerado**: revise o CSV (tipos/colunas) ou veja o código no “Ver código” e ajuste o prompt.  
- **Favoritos não atualizam**: se remover/alterar e nada acontecer, recarregue a página (o app já força rerun em remoções).

## Licença

Não definida; adicione a que preferir (ex.: MIT) se for distribuir.
