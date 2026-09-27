#!/usr/bin/env python3
"""
===============================================================================
ENEM PRO — RAG ENGINE VIA LM STUDIO (PDF VECTOR STORE v2.2)
===============================================================================
Script para ler os 2 arquivos PDF de RAG do ENEM:
- enem_rag_vectorstore_txt-1.pdf
- enem_rag_vectorstore_json-2.pdf

E realizar consultas RAG / Avaliação de Redação ENEM via LM Studio (API OpenAI Local).

Uso:
  1. Avaliar redação via CLI:
     python3 rag_lmstudio.py --essay "Texto da redação aqui..." --url "http://localhost:1234/v1"

  2. Fazer uma consulta RAG sobre as regras do ENEM:
     python3 rag_lmstudio.py --query "Como é pontuada a Competência 1 com rasura?"

  3. Iniciar um Servidor Proxy RAG Local para a Interface Web:
     python3 rag_lmstudio.py --serve --port 8080 --lmstudio "http://localhost:1234/v1"
===============================================================================
"""

import sys
import os
import json
import argparse
import urllib.request
import urllib.error
import re

try:
    import pypdf
except ImportError:
    print("[AVISO] pypdf não instalado. Instale com: pip install pypdf")

DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "local-model"

PDF_FILES = [
    "enem_rag_vectorstore_txt-1.pdf",
    "enem_rag_vectorstore_json-2.pdf"
]

JSON_BASE_FILE = "enem_rag_database.json"

def load_rag_database():
    """Carrega o banco RAG do arquivo JSON ou extrai dos PDFs caso o JSON não exista."""
    if os.path.exists(JSON_BASE_FILE):
        with open(JSON_BASE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Extração dos PDFs como fallback
    rag_chunks = {}
    for pdf_path in PDF_FILES:
        if os.path.exists(pdf_path):
            try:
                reader = pypdf.PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                
                raw_chunks = re.split(r'\[CHUNK:\s*([^\]]+)\]', text)
                for i in range(1, len(raw_chunks), 2):
                    c_id = raw_chunks[i].strip().upper()
                    c_body = raw_chunks[i+1].strip()
                    rag_chunks[c_id] = c_body
            except Exception as e:
                print(f"[ERRO] Falha ao ler {pdf_path}: {e}")
    
    if not rag_chunks:
        print("[ERRO] Nenhum arquivo PDF de RAG foi encontrado!")
        sys.exit(1)
        
    return rag_chunks

def get_relevant_chunks(query, db, comp_idx=None):
    """Busca os chunks relevantes no banco RAG baseado na consulta ou competência."""
    chunks = []
    
    # Sempre inclui a Escala Geral e Regras de Anulação
    if "ESCALA_GERAL" in db:
        chunks.append(db["ESCALA_GERAL"])
    if "CASOS_ANULACAO" in db:
        chunks.append(db["CASOS_ANULACAO"])
        
    if comp_idx:
        c_prefix = f"C{comp_idx}_"
        for k, v in db.items():
            if k.startswith(c_prefix):
                chunks.append(v)
    else:
        # Busca por palavras-chave simples
        q_lower = query.lower()
        for k, v in db.items():
            if k in ["ESCALA_GERAL", "CASOS_ANULACAO"]:
                continue
            if any(term in v.lower() or term in k.lower() for term in q_lower.split()):
                chunks.append(v)
                
    return "\n\n" + "="*60 + "\n\n".join(chunks)

def call_lm_studio(api_url, model, messages, temperature=0.2, max_tokens=2500):
    """Realiza uma chamada para a API OpenAI-compatible do LM Studio."""
    endpoint = api_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        raise RuntimeError(f"HTTP {e.code}: {err_msg}")
    except Exception as e:
        raise RuntimeError(f"Erro ao conectar com LM Studio ({endpoint}): {e}")

def evaluate_essay_with_rag(essay_text, api_url, model):
    """Avalia a redação em 5 competências aplicando o RAG dos PDFs."""
    db = load_rag_database()
    print("🚀 Iniciando Avaliação RAG via LM Studio para as 5 Competências...")
    
    scores = {}
    
    for i in range(1, 6):
        print(f"⏳ Avaliando C{i} com Contexto RAG Oficial...")
        rag_context = get_relevant_chunks(f"Competencia {i}", db, comp_idx=i)
        
        system_prompt = f"""Você é um corretor oficial especialista do ENEM.
Use ESTRITAMENTE a Base de Conhecimento RAG Oficial abaixo para avaliar a Competência {i}:

--- BASE DE CONHECIMENTO RAG ENEM (PDFs OFICIAIS) ---
{rag_context}
---------------------------------------------------

Avalie a redação do aluno e retorne um JSON no seguinte formato EXATO:
{{
  "c{i}_score": <0|40|80|120|160|200>,
  "c{i}_justificativa": "<Explicação detalhada baseada no RAG>",
  "c{i}_pos": "<Ponto positivo observado>",
  "c{i}_imp": "<Recomendação de melhoria>"
}}
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Texto da Redação:\n\"\"\"\n{essay_text}\n\"\"\""}
        ]
        
        try:
            response = call_lm_studio(api_url, model, messages)
            print(f"\n--- Resposta C{i} ---")
            print(response)
            scores[f"c{i}"] = response
        except Exception as e:
            print(f"❌ Erro ao avaliar C{i}: {e}")
            
    print("\n✅ Avaliação RAG Concluída!")
    return scores

def query_rag(query_text, api_url, model):
    """Responde a uma dúvida usando o RAG dos PDFs via LM Studio."""
    db = load_rag_database()
    context = get_relevant_chunks(query_text, db)
    
    system_prompt = f"""Você é o Assistente Virtual Oficial ENEM PRO (Professora Bia).
Responda à dúvida do usuário utilizando ESTRITAMENTE a Base de Conhecimento RAG extraída dos 2 PDFs do ENEM.

--- BASE DE CONHECIMENTO RAG ENEM ---
{context}
------------------------------------
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query_text}
    ]
    
    print(f"🔎 Consultando RAG para: '{query_text}'...")
    response = call_lm_studio(api_url, model, messages)
    print("\n💡 Resposta da Professora Bia (RAG LM Studio):")
    print(response)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ENEM PRO — Engine RAG com PDFs via LM Studio")
    parser.add_argument("--essay", type=str, help="Texto da redação para avaliação RAG")
    parser.add_argument("--query", type=str, help="Dúvida ou pergunta sobre as regras do ENEM")
    parser.add_argument("--url", type=str, default=DEFAULT_LM_STUDIO_URL, help="URL do servidor LM Studio (ex: http://localhost:1234/v1)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Nome do modelo no LM Studio")
    
    args = parser.parse_args()
    
    if args.essay:
        evaluate_essay_with_rag(args.essay, args.url, args.model)
    elif args.query:
        query_rag(args.query, args.url, args.model)
    else:
        print("Modo de uso:\n  python3 rag_lmstudio.py --query 'O que é repertório legitimado?'\n  python3 rag_lmstudio.py --essay 'Texto da redação...'\n")
