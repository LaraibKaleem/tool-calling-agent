#!/usr/bin/env python3
import sys, os, json, re, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import inference as inf_module

_TAG_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

def format_tool_call(raw_json):
    try:
        obj  = json.loads(raw_json)
        tool = obj.get("tool","?")
        args = obj.get("args",{})
        lines = [f"🔧 **Tool:** `{tool}`", "**Args:**"]
        for k, v in args.items():
            lines.append(f"  • `{k}`: `{v}`")
        return "\n".join(lines)
    except:
        return f"```json\n{raw_json}\n```"

def render_response(raw, latency_ms):
    m = _TAG_RE.search(raw)
    if m:
        tool_block = format_tool_call(m.group(1).strip())
        return f"{tool_block}\n\n*⚡ {latency_ms:.0f} ms*\n\n---\n*Raw:* `{raw}`"
    return f"{raw}\n\n*⚡ {latency_ms:.0f} ms*"

def create_app():
    import gradio as gr

    EXAMPLES = [
        "What's the weather in Tokyo?",
        "Convert 100 USD to EUR",
        "Add a dentist appointment on 2024-08-15",
        "Convert 5 kg to pounds",
        "SELECT * FROM users WHERE age > 30",
        "Tell me a joke",
        "What meetings do I have on 2024-07-04?",
        "Exchange 1000 GBP to JPY",
        ]
    def chat(user_message, chat_history, raw_history):
        if not user_message.strip():
            return "", chat_history, raw_history
        t0  = time.perf_counter()
        raw = inf_module.run(user_message, raw_history)
        ms  = (time.perf_counter() - t0) * 1000
        formatted = render_response(raw, ms)
        chat_history.append({"role": "user",      "content": user_message})
        chat_history.append({"role": "assistant",  "content": formatted})
        raw_history.append({"role": "user",        "content": user_message})
        raw_history.append({"role": "assistant",   "content": raw})
        return "", chat_history, raw_history

    # def chat(user_message, chat_history, raw_history):
    #     if not user_message.strip():
    #         return "", chat_history, raw_history
    #     t0  = time.perf_counter()
    #     raw = inf_module.run(user_message, raw_history)
    #     ms  = (time.perf_counter() - t0) * 1000
    #     formatted = render_response(raw, ms)
    #     chat_history.append((user_message, formatted))
    #     raw_history.append({"role":"user",      "content":user_message})
    #     raw_history.append({"role":"assistant",  "content":raw})
    #     return "", chat_history, raw_history

    def clear_chat():
        return [], []

    with gr.Blocks(title="Tool-Calling Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Tool-Calling Mobile Assistant\nFine-tuned Qwen2.5-0.5B | Fully Offline")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation", height=450, render_markdown=True)
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask anything... e.g. 'Weather in Paris?' or 'Convert 50 USD to EUR'",
                        label="", scale=5, container=False,
                    )
                    send_btn = gr.Button("Send ▶", variant="primary", scale=1)
                clear_btn = gr.Button("🗑 Clear", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown("### 📋 Examples")
                for ex in EXAMPLES:
                    btn = gr.Button(ex, size="sm", variant="secondary")
                    btn.click(lambda x=ex: x, outputs=[msg_box])

        raw_history_state = gr.State([])

        send_btn.click(chat, inputs=[msg_box, chatbot, raw_history_state],
                       outputs=[msg_box, chatbot, raw_history_state])
        msg_box.submit(chat, inputs=[msg_box, chatbot, raw_history_state],
                       outputs=[msg_box, chatbot, raw_history_state])
        clear_btn.click(clear_chat, outputs=[chatbot, raw_history_state])

    return demo

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port",  type=int, default=7860)
    args = ap.parse_args()
    try:
        import gradio
    except:
        os.system(f"{sys.executable} -m pip install gradio -q")
    demo = create_app()
    demo.launch(server_port=args.port, share=args.share)