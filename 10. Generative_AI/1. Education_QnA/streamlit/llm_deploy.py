#!/usr/bin/env python

import argparse
import boto3
import json
import logging
import os
import PyPDF2
import requests
import sagemaker

from bs4 import BeautifulSoup
from sagemaker.huggingface import get_huggingface_llm_image_uri
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.jumpstart.model import JumpStartModel



# ----- Setup -----
BEDROCK_REGION = 'us-west-2'

print(f'Boto3 version: {boto3.__version__}')
print(f'Region: {BEDROCK_REGION}')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


# ----- Setup Bedrock -----
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=BEDROCK_REGION
)

# ----- Helper Subroutines -----
def invoke_claude_instant(prompt, **kwargs):
    body = {
        "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
        "max_tokens_to_sample": 300,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": [
        "\\n\\nHuman:"
        ],
        "anthropic_version": "bedrock-2023-05-31"
    }
    for parameter in ['max_tokens_to_sample', 'temperature', 'top_k', 'top_p']:
        if parameter in kwargs:
            body[parameter] = kwargs[parameter]

    response = bedrock_runtime.invoke_model(
        modelId = "anthropic.claude-instant-v1",
        contentType = "application/json",
        accept = "*/*",
        body = json.dumps(body)
    )
    response_body = json.loads(response.get('body').read())
    return response_body['completion']

def generate_text_from_prompt(prompt, **kwargs):
    return invoke_claude_instant(prompt, **kwargs)


# ----- NLP Helpers -----
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

def summarize(text, seed=None, max_new_tokens=200):
    return generate_text_from_prompt(
        f"""Summarize the following content in 75 words:
            <content>{text[:10000]}</content>
            Provide only the summary and nothing else""",
        temperature = 0.2,  # Low temperature for summarization
        max_tokens_to_sample = max_new_tokens
    )

def ask(context, question, seed=None, max_new_tokens=200):
    return generate_text_from_prompt(
        f"""Given the following context <context>{context[:10000]}</context>
        answer the following question in not more than 100 words.
        If you don't know the answer, say that you don't know\n{question}""",
        temperature = 0.05,  # Low temperature for accuracy
        max_tokens_to_sample = max_new_tokens
    )

def extract_question(text, seed=None):
    return generate_text_from_prompt(
        f"""Generate a question based on the following context
        <context>{text}</context>
        The question should be no longer than 100 words.
        Provide only the question and nothing else""",
        temperature = 1.0  # Maximum temperature for creativity
    )

def create_qna_pairs(text, n, seed=None, max_new_tokens=200):
    questions = []
    answers = []

    for i in range(n):
        qn = extract_question(text, seed) if i == 0 else extract_question(text)
        questions.append(qn)
        answers.append(ask(text, qn, max_new_tokens=max_new_tokens))
    return questions, answers

def create_qna_pairs_from_paras(paragraphs, seed=None, max_new_tokens=200):
    questions = []
    answers = []

    for i, text in enumerate(paragraphs):
        qn = extract_question(text, seed) if i == 0 else extract_question(text)
        questions.append(qn)
        answers.append(ask(text, qn, max_new_tokens=max_new_tokens))
    return questions, answers



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='', help='Prompt')
    args = parser.parse_args()

    if args.prompt:
        response = generate_text_from_prompt(args.prompt)
        print(f'Response:\n{response}')
