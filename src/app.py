from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
import gradio as gr
from gradio import ChatMessage
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from challenge_2_receipts.graph import run_graph
from challenge_1_solar.agent import stream_agent

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ── Challenge 2 state ─────────────────────────────────────────────────────────
_context_cache = {}


def resolve_tier(ocr_engine: str, entity_analysis: bool, unified: bool) -> str:
    if unified:
        return "Tier 5"
    if ocr_engine == "Gemini" and entity_analysis:
        return "Tier 4"
    if ocr_engine == "EasyOCR" and entity_analysis:
        return "Tier 3"
    if ocr_engine == "Gemini":
        return "Tier 2"
    return "Tier 1"


def tier_label(tier: str) -> str:
    labels = {
        "Tier 1": "Tier 1 — Raw EasyOCR",
        "Tier 2": "Tier 2 — Gemini OCR",
        "Tier 3": "Tier 3 — EasyOCR + Entity LLM",
        "Tier 4": "Tier 4 — Gemini OCR + Entity LLM",
        "Tier 5": "Tier 5 — Unified Multimodal LLM",
    }
    return labels[tier]


def update_tier_label(ocr_engine, entity_analysis, unified):
    tier = resolve_tier(ocr_engine, entity_analysis, unified)
    return f"**Active pipeline:** {tier_label(tier)}"


def receipt_chat(message, history, image, ocr_engine, entity_analysis, unified):
    tier = resolve_tier(ocr_engine, entity_analysis, unified)
    cache_key = f"{image}_{tier}"

    if cache_key not in _context_cache:
        if image is None:
            yield "Please upload a receipt image first.", ""
            return
        result = run_graph(Path(image), tier)
        _context_cache[cache_key] = result["context"]

    context = _context_cache[cache_key]
    prompt = f"""You are answering questions about a receipt.

Receipt information:
{context}

Question: {message}"""

    response = ""
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        response += chunk.content
        yield response, context


def solar_chat(message, history, thread_id):
    yield from stream_agent(message, thread_id=thread_id)


# ── App ───────────────────────────────────────────────────────────────────────
with gr.Blocks(title="AI Labs Assessment") as app:
    gr.Markdown("# AI Labs Assessment")

    with gr.Tabs():

        # ── Challenge 1 ───────────────────────────────────────────────────────
        with gr.Tab("Challenge 1 — Solar Yield Prediction"):
            gr.Markdown("Ask questions about solar yield for Australian cities.")

            thread_id_state = gr.State("solar-default")

            gr.ChatInterface(
                fn=solar_chat,
                additional_inputs=[thread_id_state],
                chatbot=gr.Chatbot(
                    label="Solar Agent",
                    height=600,
                ),
                textbox=gr.Textbox(
                    placeholder="e.g. What is the expected yield for a 100ha farm in Sydney tomorrow?",
                    label="Ask a question",
                ),
            )

        # ── Challenge 2 ───────────────────────────────────────────────────────
        with gr.Tab("Challenge 2 — Receipt Q&A"):
            gr.Markdown("Upload a receipt and ask questions about it.")

            display_output = gr.Textbox(
                label="Extracted Information", lines=15, interactive=False, render=False
            )

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="Upload Receipt", type="filepath")

                    gr.Markdown("### Pipeline Settings")

                    unified_toggle = gr.Checkbox(
                        label="Unified Mode — Gemini handles OCR + extraction in one call",
                        value=False,
                    )
                    ocr_engine = gr.Radio(
                        choices=["EasyOCR", "Gemini"],
                        value="EasyOCR",
                        label="OCR Engine",
                        interactive=True,
                    )
                    entity_toggle = gr.Checkbox(
                        label="Entity Analysis LLM", value=False, interactive=True
                    )
                    tier_display = gr.Markdown(
                        "**Active pipeline:** Tier 1 — Raw EasyOCR"
                    )

                with gr.Column(scale=2):
                    chat_interface = gr.ChatInterface(
                        fn=receipt_chat,
                        additional_inputs=[
                            image_input,
                            ocr_engine,
                            entity_toggle,
                            unified_toggle,
                        ],
                        additional_outputs=[display_output],
                        chatbot=gr.Chatbot(label="Chat", height=500),
                        textbox=gr.Textbox(
                            placeholder="e.g. What is the total amount?",
                            label="Ask a question",
                        ),
                    )

                with gr.Column(scale=1):
                    display_output.render()

            for component in [ocr_engine, entity_toggle, unified_toggle]:
                component.change(
                    update_tier_label,
                    inputs=[ocr_engine, entity_toggle, unified_toggle],
                    outputs=tier_display,
                )

            def toggle_unified(unified):
                return (
                    gr.Radio(interactive=not unified),
                    gr.Checkbox(interactive=not unified),
                )

            unified_toggle.change(
                toggle_unified,
                inputs=unified_toggle,
                outputs=[ocr_engine, entity_toggle],
            )

            def reset():
                _context_cache.clear()
                return [], ""

            image_input.change(
                reset, outputs=[chat_interface.chatbot_value, display_output]
            )
            ocr_engine.change(
                reset, outputs=[chat_interface.chatbot_value, display_output]
            )
            entity_toggle.change(
                reset, outputs=[chat_interface.chatbot_value, display_output]
            )
            unified_toggle.change(
                reset, outputs=[chat_interface.chatbot_value, display_output]
            )


if __name__ == "__main__":
    app.launch()
