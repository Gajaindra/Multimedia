from transformers import pipeline

# Load once globally
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def answer_question(context, question):
    if not context or not question:
        return "Please provide both context and question."
    result = qa_pipeline(question=question, context=context)
    return result["answer"]
