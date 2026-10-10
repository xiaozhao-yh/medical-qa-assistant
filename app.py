import os
import json
import time
import numpy as np
import PyPDF2
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer
from typing import List

import auth_utils

DATA_DIR = "data"
DOCS_DIR = "docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
DASHSCOPE_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    import openai
    openai.api_key = OPENAI_API_KEY

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

embed_model = SentenceTransformer(EMBED_MODEL_NAME)

metadatas = []
embeddings_np = None
faiss_index = None
use_faiss = False

def ensure_index_loaded():
    global metadatas, embeddings_np, faiss_index, use_faiss
    meta_path = os.path.join(DATA_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadatas = json.load(f)
    else:
        metadatas = []

    try:
        import faiss
        index_path = os.path.join(DATA_DIR, "index.faiss")
        if os.path.exists(index_path):
            faiss_index = faiss.read_index(index_path)
            use_faiss = True
            print("已加载 FAISS 索引。")
        else:
            raise Exception("index.faiss 不存在")
    except Exception:
        emb_path = os.path.join(DATA_DIR, "embeddings.npy")
        if os.path.exists(emb_path):
            embeddings_np = np.load(emb_path)
        else:
            embeddings_np = np.zeros((0, embed_model.get_sentence_embedding_dimension()), dtype="float32")
        use_faiss = False

ensure_index_loaded()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
        if start >= length:
            break
    return [c for c in chunks if c]


def persist_index():
    global metadatas, embeddings_np, faiss_index, use_faiss
    with open(os.path.join(DATA_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

    if use_faiss and faiss_index is not None:
        import faiss
        faiss.write_index(faiss_index, os.path.join(DATA_DIR, "index.faiss"))
    else:
        if embeddings_np is not None:
            np.save(os.path.join(DATA_DIR, "embeddings.npy"), embeddings_np)


def retrieve(query: str, top_k: int = TOP_K):
    global embeddings_np, faiss_index, metadatas, use_faiss

    if not metadatas:
        return []

    q_emb = embed_model.encode(query).astype("float32")

    if use_faiss and faiss_index is not None:
        _, I = faiss_index.search(np.array([q_emb]), top_k)
        indices = I[0].tolist()
    else:
        if embeddings_np is None or embeddings_np.shape[0] == 0:
            return []
        sims = np.dot(embeddings_np, q_emb)
        indices = np.argsort(-sims)[:top_k].tolist()

    results = []
    for rank, idx in enumerate(indices):
        if idx < 0 or idx >= len(metadatas):
            continue
        m = metadatas[idx]
        results.append({
            "score_rank": rank + 1,
            "source": m.get("source"),
            "chunk_id": m.get("chunk_id"),
            "text": m.get("text")
        })
    return results


def add_document_from_text(text: str, source_name: str = "uploaded", persist: bool = True):
    global metadatas, embeddings_np, faiss_index, use_faiss

    chunks = chunk_text(text)
    if not chunks:
        return 0

    embs = embed_model.encode(chunks).astype("float32")
    for i, c in enumerate(chunks):
        metadatas.append({"source": source_name, "chunk_id": i, "text": c})

    if use_faiss and faiss_index is not None:
        import faiss
        faiss_index.add(embs)
    else:
        if embeddings_np is None or embeddings_np.size == 0:
            embeddings_np = embs
        else:
            embeddings_np = np.vstack([embeddings_np, embs])

    if persist:
        persist_index()
    return len(chunks)


def extract_text_from_file(file_path: str):
    _, ext = os.path.splitext(file_path.lower())
    if ext in (".txt", ".md"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            try:
                with open(file_path, "r", encoding="gbk", errors="ignore") as f:
                    return f.read()
            except Exception:
                return ""

    elif ext == ".pdf":
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = []
                for page in reader.pages:
                    text = page.extract_text() or ""
                    pages.append(text)
                return "\n".join(pages)
        except Exception:
            return ""

    return ""


def build_prompt(query: str, contexts: List[dict], language: str, style: str):
    if language == "English":
        disclaimer = "This system is for learning and reference only and does NOT constitute medical advice."
        system_intro = "You are a medical knowledge assistant. Answer based only on the provided reference materials."
        answer_lang = "English"
        if style == "short":
            style_text = "Brief and concise. Provide the answer in a few bullet points."
        else:
            style_text = "Detailed and structured answer with explanations, cautions, and steps if appropriate."
    else:
        disclaimer = "本系统仅供学习参考，不构成医疗建议。"
        system_intro = "你是一个医疗知识助手。请仅基于下列参考资料进行回答。"
        answer_lang = "中文"
        if style == "short":
            style_text = "简短且要点突出，尽量控制在 3-6 条要点。"
        else:
            style_text = "详细而结构化，说明原因、注意事项和必要的就医建议。"

    ctx_texts = []
    for i, c in enumerate(contexts, start=1):
        if language == "English":
            header = f"[Source{i}] file: {c['source']} chunk: {c['chunk_id']}"
        else:
            header = f"[来源{i}] 文件: {c['source']} 段落: {c['chunk_id']}"
        ctx_texts.append(f"{header}\n{c['text']}")
    ctx_section = "\n\n".join(ctx_texts)

    system_prompt = f"""
{system_intro}
{disclaimer}
Answer language: {answer_lang}
Answer style: {style_text}
Please explicitly cite the source numbers in the answer like [来源1], [Source1].
If the answer is uncertain, say so honestly and recommend consulting a doctor or healthcare professional.
"""

    user_message = f"""
User question:
{query}

Reference materials:
{ctx_section}
"""

    return system_prompt, user_message


def ask_qwen_bailian(system_prompt: str, user_message: str, max_tokens=512, temperature=0.2):
    if not DASHSCOPE_API_KEY:
        return "未配置 DASHSCOPE_API_KEY，无法调用阿里云百炼 Qwen。"

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DASHSCOPE_MODEL,
        "input": {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        },
        "parameters": {
            "result_format": "message",
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    }

    try:
        resp = requests.post(DASHSCOPE_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if "output" in data and "choices" in data["output"]:
            content = data["output"]["choices"][0]["message"]["content"]
            return content.strip()

        if "output" in data and "text" in data["output"]:
            return data["output"]["text"].strip()

        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"百炼 Qwen 调用失败：{e}"


def ask_openai_fallback(system_prompt: str, user_message: str, max_tokens=512, temperature=0.2):
    if not OPENAI_API_KEY:
        return "未配置 OPENAI_API_KEY，且未配置 DASHSCOPE_API_KEY。"

    try:
        import openai
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI 回退调用失败：{e}"


def generate_answer(system_prompt: str, user_message: str, max_tokens=512, temperature=0.2):
    if DASHSCOPE_API_KEY:
        return ask_qwen_bailian(system_prompt, user_message, max_tokens=max_tokens, temperature=temperature)
    elif OPENAI_API_KEY:
        return ask_openai_fallback(system_prompt, user_message, max_tokens=max_tokens, temperature=temperature)
    else:
        return "未配置大模型 API Key。请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY。"


with gr.Blocks(title="医疗知识问答助手（增强版）") as demo:
    gr.Markdown("# 🏥 医疗知识问答助手（增强版）")
    gr.Markdown("提示：本系统仅供学习与参考，不构成医疗建议。")

    with gr.Column() as login_col:
        gr.Markdown("## 登录 / Login")
        username_in = gr.Textbox(label="用户名 / Username")
        password_in = gr.Textbox(label="密码 / Password", type="password")
        login_btn = gr.Button("登录 / Login")
        login_msg = gr.Markdown("")

    with gr.Column(visible=False) as main_col:
        with gr.Row():
            lang = gr.Dropdown(choices=["中文", "English"], value="中文", label="界面语言 / UI Language")
            style = gr.Radio(choices=[("简短 / Short", "short"), ("详细 / Detailed", "detailed")], value="short", label="回答风格 / Answer Style")

        with gr.Row():
            query = gr.Textbox(lines=2, placeholder="请输入你的问题 / Ask your question here")
            submit = gr.Button("提问 / Ask")

        with gr.Row():
            upload_files = gr.File(label="上传病历/文档（.txt/.md/.pdf）", file_count="multiple")
            upload_btn = gr.Button("上传并加入索引 / Upload and index")
            reindex_btn = gr.Button("重建索引 / Rebuild index")

        chat = gr.Chatbot()
        retrieved_box = gr.Markdown("")
        status_md = gr.Markdown("")

    def do_login(u, p):
        if not u or not p:
            return gr.update(visible=True), gr.update(visible=False), "请填写用户名与密码。"
        ok = auth_utils.verify_user(u, p)
        if ok:
            return gr.update(visible=False), gr.update(visible=True), f"登录成功，欢迎：{u}"
        return gr.update(visible=True), gr.update(visible=False), "用户名或密码错误。"

    login_btn.click(do_login, [username_in, password_in], [login_col, main_col, login_msg])

    def on_upload(files):
        if not files:
            return "未上传文件。"
        info_lines = []
        for f in files:
            file_name = os.path.basename(f.name)
            text = extract_text_from_file(f.name)
            if not text or len(text.strip()) == 0:
                info_lines.append(f"{file_name}: 无法提取文本或文件为空。")
                continue

            source_name = f"uploaded_{int(time.time())}_{file_name}"
            n = add_document_from_text(text, source_name=source_name, persist=True)
            info_lines.append(f"{file_name}: 添加到索引，分片数 {n}")

        return "\n\n".join(info_lines)

    upload_btn.click(on_upload, [upload_files], status_md)

    def do_reindex():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("build_index_module", "build_index.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.build_and_save()
            ensure_index_loaded()
            return "索引重建已完成。若未生效，请重启服务。"
        except Exception as e:
            return f"重建索引失败：{e}"

    reindex_btn.click(do_reindex, [], status_md)

    def on_ask(q, hist, lang_sel, style_sel):
        if not q or q.strip() == "":
            return hist, "请先输入问题。"

        contexts = retrieve(q, top_k=TOP_K)
        system_prompt, user_message = build_prompt(q, contexts, language=lang_sel, style=style_sel)

        max_tokens = 220 if style_sel == "short" else 600
        temperature = 0.15 if style_sel == "short" else 0.25

        answer = generate_answer(system_prompt, user_message, max_tokens=max_tokens, temperature=temperature)

        sources_md_lines = []
        for i, c in enumerate(contexts, start=1):
            preview = c["text"][:400].replace("\n", " ")
            sources_md_lines.append(
                f"**[来源{i}]** `{c['source']}` — 段 {c['chunk_id']}\n\n> {preview}{'...' if len(c['text']) > 400 else ''}"
            )

        sources_md = "\n\n".join(sources_md_lines) if sources_md_lines else "未检索到参考资料。"

        hist = hist or []
        hist.append((q, answer))
        return hist, sources_md

    submit.click(on_ask, [query, chat, lang, style], [chat, retrieved_box])

if __name__ == "__main__":
    demo.launch()
