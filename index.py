import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Corretor ENEM RTX 4090", layout="wide")

# --- CABEÇALHO ---
st.title("📝 Corretor de Redação ENEM")
st.subheader("Processamento Local: NVIDIA RTX 4090")

# --- CONFIGURAÇÃO DAS URLs (ATUALIZE AQUI!) ---
# 1. Rode 'ngrok http 1234' no terminal
# 2. Copie o link e cole abaixo (mantenha o /v1 no final)
URL_DA_IA = "https://carolann-unfrosty-semiliberally.ngrok-free.dev/v1"

client = OpenAI(
    base_url=URL_DA_IA,
    api_key="lm-studio",
    # Esta linha abaixo é essencial para funcionar em outros PCs/Celular
    default_headers={"ngrok-skip-browser-warning": "true"}
)

# --- LÓGICA DO CORRETOR ---
system_instruction = (
    "Você é um corretor rigoroso do ENEM. Avalie a redação seguindo as 5 competências do INEP. "
    "Dê notas de 0 a 200 para cada uma e justifique. No final, dê a nota total."
)

texto_redacao = st.text_area("Cole sua redação completa aqui:", height=350)

if st.button("🚀 Iniciar Correção Profissional"):
    if texto_redacao:
        with st.spinner("Sua RTX 4090 está analisando a redação..."):
            try:
                chat_completion = client.chat.completions.create(
                    model="local-model",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": texto_redacao}
                    ],
                    temperature=0.3
                )
                
                resposta = chat_completion.choices[0].message.content
                st.markdown("---")
                st.markdown(resposta)
                
            except Exception as e:
                st.error(f"Erro de conexão: {e}")
                st.info("Verifique se o terminal do Ngrok (porta 1234) e o LM Studio estão ativos.")
    else:
        st.warning("Por favor, insira o texto da redação.")