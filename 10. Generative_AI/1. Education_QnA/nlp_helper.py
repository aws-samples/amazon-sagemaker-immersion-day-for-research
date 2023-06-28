import pandas as pd
import PyPDF2
import re
import requests
import json
import boto3
from IPython.display import display, HTML, IFrame

from bs4 import BeautifulSoup

pd.set_option('max_colwidth', 80)  # Set max column width for displaying Pandas Dataframes
QNA_OUTPUT_STYLE = 'HTML'

def extract_pages(pdf_file, max_pages=100):
    pages = []
    with open(pdf_file, 'rb') as f:
        for i, page in enumerate(PyPDF2.PdfReader(f).pages):
            if i == max_pages:
                break
            pages.append(page.extract_text())
    return pages

def download_url_text(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.content.decode('utf-8')
    else:
        print(f'Failed to download {url}. Status code = {r.status_code}')
        return None
    
def extract_paragraphs_from_html(text):
    html = BeautifulSoup(text, 'html.parser')
    return [ p.text for p in html.body.select('p') ]


newline, bold, unbold = '\n', '\033[1m', '\033[0m'
lightred, lightgreen, lightyellow, lightblue = '\033[91m', '\033[92m', '\033[93m', '\033[94m'
lightmagenta, lightcyan, reset = '\033[95m', '\033[96m', '\33[39m'

endpoint_name = ""
def query_endpoint_with_json_payload(encoded_json):
    client = boto3.client('runtime.sagemaker')
    response = client.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=encoded_json)
    return response

def parse_response_multiple_texts(query_response):
    model_predictions = json.loads(query_response['Body'].read())
    generated_text = model_predictions['generated_texts']
    return generated_text

def generate_text_from_prompt(prompt, max_length=300, max_time=50, temperature=0.5,
                              top_k=None, top_p=None, do_sample=True, seed=None):
    payload = {
        "text_inputs": prompt,
        "max_length": max_length,
        "max_time": max_time,
        "temperature": temperature,
        "do_sample": do_sample
    }
    if top_k is not None:
        payload['top_k'] = top_k
    if top_p is not None:
        payload['top_p'] = top_p
    if seed is not None:
        payload['seed'] = seed

    query_response = query_endpoint_with_json_payload(json.dumps(payload).encode('utf-8'))
    return parse_response_multiple_texts(query_response)[0]

def summarize(text, seed=None):
    return generate_text_from_prompt(
        f"""Summarize the following text in 100 words:\n\n{text}\n\nSummary:""",
        temperature=0.2,  # Low temperature for summarization
        seed=seed
    )

def ask(context, question, seed=None):
    return generate_text_from_prompt(
        f"""CONTEXT:\n{context}\n{question}""",
        temperature=0.01,  # Lowest temperature for accuracy
        max_length=150,    # Keep answers from being too verbose
        seed=seed
    )

def extract_question(text, seed=None):
    return generate_text_from_prompt(
        f"""EXTRACT QUESTIONS\nContext:\n{text}\nQuestion:""",
        temperature=1.0,  # Maximum temperature for creativity
        seed=seed
    )

def create_qna_pairs(text, n, output_style='HTML', seed=None):
    questions = []
    answers = []

    for i in range(n):
        qn = extract_question(text, seed) if i == 0 else extract_question(text)
        questions.append(qn)
        answers.append(ask(text, qn))
        if output_style == 'HTML':
            output = \
            f"""<b>{i+1}</b>. <b><font color=#FF7F50>Question</font></b>: {questions[i]}
            <b><font color=#FA8072>Answer</font></b>: {answers[i]}"""
            display(HTML(output))
        elif output_style == 'text':
            print(f"""{i+1}. {lightblue}{bold}Question{unbold}{reset}: {questions[i]} {lightcyan}{bold}Answer{unbold}{reset}: {answers[i]}""")
    if output_style == 'table':
        return pd.DataFrame({
            'Question': questions,
            'Answer': answers
        }).drop_duplicates()