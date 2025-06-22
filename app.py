import gradio as gr
import os
from image_module import caption_image, image_qa
from pdf_module import extract_text_from_pdf
from text_qa_module import answer_question

def image_caption_fn(image):
    return caption_image(image)

def image_qa_fn(image, question):
    return image_qa(image, question)

def pdf_qa_fn(pdf_file, question):
    text = extract_text_from_pdf(pdf_file)
    if not text.strip():
        return "No text extracted from PDF."
    return answer_question(text, question)

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 SmartVision - Multi-Modal AI Assistant")

    with gr.Tab("Image Captioning"):
        img = gr.Image(type="pil")
        btn = gr.Button("Generate Caption")
        output = gr.Textbox(label="Caption")
        btn.click(image_caption_fn, inputs=img, outputs=output)

    with gr.Tab("Image Question Answering"):
        img2 = gr.Image(type="pil")
        question2 = gr.Textbox(label="Ask a question about the image")
        btn2 = gr.Button("Get Answer")
        output2 = gr.Textbox(label="Answer")
        btn2.click(image_qa_fn, inputs=[img2, question2], outputs=output2)

    with gr.Tab("PDF Question Answering"):
        pdf = gr.File(label="Upload PDF")
        question3 = gr.Textbox(label="Ask a question about the PDF")
        btn3 = gr.Button("Get Answer")
        output3 = gr.Textbox(label="Answer")
        btn3.click(pdf_qa_fn, inputs=[pdf, question3], outputs=output3)

demo.launch()
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)