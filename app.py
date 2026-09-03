import os 
import tempfile
import uuid

import streamlit as st

from setup_db import create_database, DB_PATH

# Ensure DB exists (needed on first deploy to Streamlit Cloud)
if not os.path.exists(DB_PATH):
    create_database()

from shopping_agent import agent, set_session_id


# ---------------------------------------------------------------------------
# Agent trace helpers — surface the tool calls behind each answer
# ---------------------------------------------------------------------------
def extract_tool_trace(result) -> list:
    """
    Walk the agent's message list and pair each tool call (name + arguments) with the
    result it produced. Returns a list of steps: {name, args, output}.
    """
    messages = result.get("messages", []) if isinstance(result, dict) else []

    # Collect tool outputs keyed by the tool_call_id they respond to.
    outputs_by_id = {}
    for msg in messages:
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id is not None:
            outputs_by_id[tool_call_id] = getattr(msg, "content", "")

    steps = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            # tool_calls entries can be dicts (LangChain) with name/args/id.
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "?")
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            steps.append(
                {
                    "name": name,
                    "args": args or {},
                    "output": outputs_by_id.get(call_id, ""),
                }
            )
    return steps


def render_tool_trace(trace: list) -> None:
    """Render a collapsible view of which tools the agent used for an answer."""
    if not trace:
        return
    label = f"🔍 Show what the agent did ({len(trace)} tool call{'s' if len(trace) != 1 else ''})"
    with st.expander(label, expanded=False):
        for i, step in enumerate(trace, start=1):
            args_str = ", ".join(f"{k}={v!r}" for k, v in step["args"].items())
            st.markdown(f"**{i}. Called** `{step['name']}({args_str})`")
            output = str(step["output"])
            if len(output) > 1500:
                output = output[:1500] + " …(truncated)"
            st.code(output or "(no output)", language="json")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Shopping Agent", page_icon="🛒", layout="wide")

st.title("🛒 AI Shopping Assistant")
st.caption("Tell me what you want — I'll search, rate, and order the best match for you.")

# ---------------------------------------------------------------------------
# Sidebar — shop by image
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Shop by Image")
    st.caption("Upload a photo of a product and I'll find similar items in our store.")

    uploaded_file = st.file_uploader(
        "Upload product image", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

    if uploaded_file and st.button("Find similar products", use_container_width=True):
        suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            image_path = tmp.name

        prompt = f"I uploaded a product image. Please analyze it and find similar products in the store. Image path: {image_path}"
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_image = uploaded_file.name
        st.rerun()

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Unique id for this browser session, used to scope orders to this shopper.
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

# Render history — show a friendlier label for image-search messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and msg["content"].startswith("I uploaded a product image"):
            filename = msg["content"].split("Image path:")[-1].strip()
            st.markdown(f"Searching by image: **{os.path.basename(filename)}**")
        else:
            st.markdown(msg["content"].replace("$", r"\$"))
        if msg["role"] == "assistant" and msg.get("trace"):
            render_tool_trace(msg["trace"])

# ---------------------------------------------------------------------------
# Run agent if there's an unprocessed message (image upload triggers this)
# ---------------------------------------------------------------------------
if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
    and "pending_image" in st.session_state
):
    with st.chat_message("assistant"):
        with st.spinner("Analyzing image and searching…"):
            set_session_id(st.session_state.session_id)
            result = agent.invoke(
                {"messages": st.session_state.messages},
                config={"recursion_limit": 50},
            )
            response = result["messages"][-1].content.replace("`", "")
            trace = extract_tool_trace(result)
        st.markdown(response.replace("$", r"\$"))
        render_tool_trace(trace)

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "trace": trace}
    )
    del st.session_state.pending_image
    st.rerun()

# ---------------------------------------------------------------------------
# Text input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("e.g. I want organic honey under $15 with 4+ rating"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            set_session_id(st.session_state.session_id)
            result = agent.invoke(
                {"messages": st.session_state.messages},
                config={"recursion_limit": 50},
            )
            response = result["messages"][-1].content.replace("`", "")
            trace = extract_tool_trace(result)
        st.markdown(response.replace("$", r"\$"))
        render_tool_trace(trace)

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "trace": trace}
    )
    st.rerun()