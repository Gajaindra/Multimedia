import gradio as gr
import os
from image_module import caption_image
from pdf_module import extract_text_from_pdf
from text_qa_module import answer_question

def pdf_qa_fn(pdf_file, question):
    text = extract_text_from_pdf(pdf_file)
    if not text.strip():
        return "No text extracted from PDF."
    
    # Simple chunking
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    for chunk in chunks:
        try:
            answer = answer_question(chunk, question)
            if answer:
                return answer
        except:
            continue
    return "No good answer found."

# Get port from environment for Render
port = int(os.environ.get("PORT", 7860))

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 SmartVision - Multi-Modal AI Assistant")

    with gr.Tab("📷 Image Captioning"):
        img = gr.Image(type="pil")
        btn = gr.Button("Generate Caption")
        output = gr.Textbox(label="Caption")
        btn.click(caption_image, inputs=img, outputs=output)

    with gr.Tab("📄 PDF Question Answering"):
        pdf = gr.File(label="Upload PDF")
        question = gr.Textbox(label="Ask a question about the PDF")
        btn2 = gr.Button("Get Answer")
        output2 = gr.Textbox(label="Answer")
        btn2.click(pdf_qa_fn, inputs=[pdf, question], outputs=output2)

demo.launch(server_name="0.0.0.0", server_port=port)
