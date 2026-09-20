import os

import gradio as gr
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert email marketing copywriter. Write clear, persuasive, "
            "specific copy that matches the requested audience, campaign, and tone. "
            "Return exactly these sections with the labels shown, and do not add "
            "anything outside them:\n\n"
            "SUBJECT LINE 1:\n"
            "SUBJECT LINE 2:\n"
            "SUBJECT LINE 3:\n"
            "PREVIEW TEXT:\n"
            "MARKETING EMAIL:\n"
            "CTA:\n"
            "SHORT VERSION:\n\n"
            "Keep subject lines under 60 characters when possible. Make the email "
            "easy to scan with short paragraphs. Never invent guarantees, prices, "
            "or product claims that were not provided.",
        ),
        (
            "human",
            "Product or service: {product}\n"
            "Target audience: {audience}\n"
            "Campaign type: {campaign_type}\n"
            "Tone: {tone}\n"
            "Key features or benefits: {features}\n"
            "Offer: {offer}\n"
            "Desired CTA: {cta}\n\n"
            "Create the requested email marketing content.",
        ),
    ]
)


def generate_content(
    product: str,
    audience: str,
    campaign_type: str,
    tone: str,
    features: str,
    offer: str,
    cta: str,
) -> str:
    """Generate email marketing content from the form values."""
    if not os.getenv("GROQ_API_KEY"):
        raise gr.Error("Set GROQ_API_KEY in the Space secrets or your .env file first.")

    model = ChatGroq(
        model=MODEL_NAME,
        temperature=0.7,
        max_tokens=1800,
    )
    chain = PROMPT | model | StrOutputParser()

    try:
        return chain.invoke(
            {
                "product": product.strip(),
                "audience": audience.strip(),
                "campaign_type": campaign_type,
                "tone": tone,
                "features": features.strip(),
                "offer": offer.strip() or "No specific offer was provided.",
                "cta": cta.strip(),
            }
        )
    except Exception as error:
        raise gr.Error(f"Groq request failed: {error}") from error


with gr.Blocks(title="AI Email Marketing Agent") as demo:
    gr.Markdown("# AI Email Marketing Agent\nGenerate campaign-ready email copy with Groq and Llama.")

    with gr.Row():
        with gr.Column(scale=1):
            product = gr.Textbox(label="Product or service", placeholder="e.g. Project management app")
            audience = gr.Textbox(label="Target audience", placeholder="e.g. Small creative teams")
            campaign_type = gr.Dropdown(
                ["Product launch", "Promotional", "Welcome", "Newsletter", "Re-engagement", "Event invitation"],
                value="Promotional",
                label="Campaign type",
            )
            tone = gr.Dropdown(
                ["Friendly", "Professional", "Playful", "Urgent", "Helpful", "Bold"],
                value="Friendly",
                label="Tone",
            )
            features = gr.Textbox(
                label="Key features or benefits",
                lines=4,
                placeholder="List the details that make the offer valuable",
            )
            offer = gr.Textbox(label="Offer", placeholder="e.g. 20% off through Friday")
            cta = gr.Textbox(label="CTA", value="Get started today")
            generate = gr.Button("Generate email content", variant="primary")

        with gr.Column(scale=1):
            output = gr.Markdown(label="Generated content")

    generate.click(
        fn=generate_content,
        inputs=[product, audience, campaign_type, tone, features, offer, cta],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch()
