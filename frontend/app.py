import os

import requests
import streamlit as st


API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
SUPPORTED_TYPES = [
    "pdf",
    "docx",
    "xlsx",
    "xls",
    "html",
    "htm",
    "csv",
    "txt",
    "md",
    "markdown",
]
WELCOME = (
    "Ask a question about your indexed documents. Answers are grounded in "
    "authorized sources and include citations."
)

st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="K",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink: #0f172a; --muted: #64748b; }
    .stApp { background: #ffffff; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { display: none; }
    .block-container {
        max-width: 820px;
        padding-top: 1.5rem;
        padding-bottom: 7rem;
    }
    html, body, [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li { color: var(--ink); }
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 { color: var(--ink); }
    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.35rem 0;
    }
    .welcome-title {
        text-align: center;
        font-size: 1.9rem;
        font-weight: 650;
        color: var(--ink);
        margin: 3.5rem 0 0.4rem 0;
    }
    .welcome-sub {
        text-align: center;
        color: var(--muted);
        margin-bottom: 1.4rem;
    }
    [data-testid="stChatInput"] textarea {
        color: var(--ink) !important;
    }
    .stButton>button {
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background: #f7f8fa;
        border-right: 1px solid #eceff3;
    }
    [data-testid="stSidebar"] * { color: var(--ink); }
    [data-testid="stFileUploaderDropzone"] {
        background: #fff;
        border: 1px dashed #c3ccd8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30, show_spinner=False)
def load_config() -> dict:
    response = requests.get(f"{API_URL}/api/v1/access-tags", timeout=5)
    response.raise_for_status()
    return response.json()


try:
    app_config = load_config()
except requests.RequestException:
    st.error(
        "The local API is unavailable. Start the FastAPI backend and refresh "
        "this page."
    )
    st.stop()

access_tags = app_config["access_tags"]
max_upload_size_mb = app_config.get("max_upload_size_mb", 25)


def format_citation(citation: dict) -> str:
    section = " > ".join(citation.get("section") or []) or "Section unavailable"
    pages = ", ".join(map(str, citation.get("pages") or []))
    return f"{section} | Pages {pages}" if pages else f"{section} | Page unavailable"


def ask_api(question: str) -> dict:
    response = requests.post(
        f"{API_URL}/api/v1/query",
        json={"question": question},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def render_assistant(payload: dict) -> None:
    st.markdown(payload["answer"])
    if payload.get("not_found"):
        st.caption("No supporting evidence was found in the indexed documents.")
        return
    citations = payload.get("citations") or []
    if citations:
        with st.expander(f"Sources ({len(citations)})"):
            for citation in citations:
                st.markdown(
                    f"**[{citation['reference']}] {citation['source_file']}**  \n"
                    f"{format_citation(citation)}"
                )


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### Knowledge Assistant")
    st.caption("Local, cited answers from your own documents.")
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("pending_question", None)
        st.rerun()

    st.divider()
    with st.expander("Add documents", expanded=False):
        st.caption(
            f"PDF, Word, Excel, HTML, CSV, text, or Markdown up to "
            f"{max_upload_size_mb} MB."
        )
        department = st.selectbox("Access department", access_tags)
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=SUPPORTED_TYPES,
            label_visibility="collapsed",
        )
        upload_too_large = bool(
            uploaded_file
            and uploaded_file.size > max_upload_size_mb * 1024 * 1024
        )
        if upload_too_large:
            st.error(f"Choose a file smaller than {max_upload_size_mb} MB.")
        if st.button(
            "Upload and index",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None or upload_too_large,
        ):
            with st.spinner("Parsing, embedding, and indexing..."):
                try:
                    response = requests.post(
                        f"{API_URL}/api/v1/documents/upload",
                        data={"department": department},
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type
                                or "application/octet-stream",
                            )
                        },
                        timeout=1800,
                    )
                    if response.ok:
                        result = response.json()
                        st.success(
                            f"Indexed {result['chunk_count']} chunks. "
                            f"ID: {result['document_id']}"
                        )
                        if result.get("parser_warnings"):
                            st.warning(
                                "Parser issues: "
                                + ", ".join(result["parser_warnings"])
                            )
                    else:
                        st.error(f"Upload failed: {response.text}")
                except requests.RequestException as exc:
                    st.error(f"Could not reach the local API: {exc}")

    with st.expander("Check processing status", expanded=False):
        document_id = st.text_input("Document ID", key="status_document_id")
        if st.button(
            "Check status",
            use_container_width=True,
            disabled=not document_id.strip(),
        ):
            try:
                response = requests.get(
                    f"{API_URL}/api/v1/documents/{document_id.strip()}",
                    timeout=10,
                )
                response.raise_for_status()
                result = response.json()
                st.write(
                    {
                        "filename": result["original_filename"],
                        "status": result["processing_status"],
                        "uploaded": result["uploaded_at"],
                        "version": result["current_version"],
                        "parser_warnings": result.get("parser_warnings", []),
                        "failure_reason": result.get("failure_reason"),
                    }
                )
            except requests.HTTPError as exc:
                st.error(f"Status lookup failed: {exc.response.text}")
            except requests.RequestException as exc:
                st.error(f"Could not reach the local API: {exc}")


if not st.session_state.messages:
    st.markdown('<div class="welcome-title">How can I help you today?</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="welcome-sub">{WELCOME}</div>', unsafe_allow_html=True)
    suggestions = [
        "What is the annual leave entitlement?",
        "What are the hotel limits per night?",
        "What expenses are not reimbursed?",
    ]
    columns = st.columns(len(suggestions))
    for column, suggestion in zip(columns, suggestions):
        if column.button(suggestion, use_container_width=True):
            st.session_state.pending_question = suggestion
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant(message["payload"])
        else:
            st.markdown(message["content"])

prompt = st.chat_input("Message the knowledge assistant...")
if "pending_question" in st.session_state:
    prompt = st.session_state.pop("pending_question")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching authorized sources..."):
            try:
                payload = ask_api(prompt)
            except requests.HTTPError as exc:
                payload = {
                    "answer": f"Query failed: {exc.response.text}",
                    "not_found": True,
                    "citations": [],
                }
            except requests.RequestException as exc:
                payload = {
                    "answer": f"Could not reach the local API: {exc}",
                    "not_found": True,
                    "citations": [],
                }
        render_assistant(payload)
    st.session_state.messages.append(
        {"role": "assistant", "payload": payload}
    )
