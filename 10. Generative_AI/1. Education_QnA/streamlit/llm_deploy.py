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
DO_SAMPLE = True
TOP_P = 0.9
TEMPERATURE = 0.8
# MIN_LENGTH = 25
# MAX_LENGTH = 200
MAX_NEW_TOKENS = 1024
NO_REPEAT_NGRAM_SIZE = None
ENDPOINT_NAME = 'None'
REGION = os.getenv('AWS_DEFAULT_REGION', default='us-east-1')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

valid_model_ids = [
    'huggingface-llm-falcon-40b-bf16',
    'huggingface-llm-falcon-40b-instruct-bf16',
    'huggingface-llm-falcon-7b-bf16',
    'huggingface-llm-falcon-7b-instruct-bf16',
    'huggingface-textgeneration1-redpajama-incite-instruct-7B-v1-fp16'
    ]

valid_hf_model_ids = [
    'google/flan-ul2',
    'google/flan-t5-xl',
    'google/flan-t5-xxl',
    'tiiuae/falcon-7b',
    'tiiuae/falcon-7b-instruct',
    'tiiuae/falcon-40b',
    'tiiuae/falcon-40b-instruct',
    'togethercomputer/RedPajama-INCITE-7B-Base',
    'togethercomputer/RedPajama-INCITE-7B-Chat',
    'togethercomputer/RedPajama-INCITE-7B-Instruct'
    ]


# ----- Setup SageMaker -----
try:
	role = sagemaker.get_execution_role()
except ValueError:
	iam = boto3.client('iam')
	role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

client = boto3.client(
    'sagemaker',
    region_name = REGION,
)

# ----- Helper Subroutines -----
def create_payload_from_prompt0(prompt):
    global DO_SAMPLE
    global TOP_P
    global TEMPERATURE
    global MAX_NEW_TOKENS
    global NO_REPEAT_NGRAM_SIZE

    parameters = {
        "do_sample": DO_SAMPLE,
        "top_p": TOP_P,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "stop": ["<|endoftext|>", "</s>"]
        }

    if NO_REPEAT_NGRAM_SIZE is not None:
        parameters['no_repeat_ngram_size'] = NO_REPEAT_NGRAM_SIZE

    payload = {
        "inputs": prompt,
        "parameters": parameters
    }
    return payload


def create_payload_from_prompt_flan(
        prompt, max_length=300, max_time=50, temperature=0.5,
        top_k=None, top_p=None, do_sample=True, seed=None, max_new_tokens=None):

    if max_new_tokens is not None:
        print('Warning: max_new_tokens={max_new_tokens} will be ignored by Flan')

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

    return payload


def create_payload_from_prompt_redpajama(
        prompt, max_length=300, temperature=1.0, num_return_sequences=1,
        top_k=250, top_p=0.8, do_sample=True, seed=None):
    parameters = {
        "max_length": max_length,
    }

    if temperature is not None:
        parameters['temperature'] = temperature
    if do_sample is not None:
        parameters['do_sample'] = do_sample
    if num_return_sequences is not None:
        parameters['num_return_sequences'] = num_return_sequences
    if max_length is not None:
        parameters['max_length'] = max_length
    if top_k is not None:
        parameters['top_k'] = top_k
    if top_p is not None:
        parameters['top_p'] = top_p
    if seed is not None:
        parameters['seed'] = seed

    payload = {
        "text_inputs": prompt,
        "parameters": parameters
    }
    return payload


def create_payload_from_prompt_falcon(
        prompt, temperature=0.8, max_new_tokens=300, max_time=None,
        top_k=None, top_p=0.9, no_repeat_ngram_size=4,
        stop=["<|endoftext|>", "</s>"],
        do_sample=True, seed=None):
    parameters = {
        }

    if temperature is not None:
        parameters['temperature'] = temperature
    if do_sample is not None:
        parameters['do_sample'] = do_sample
    if stop is not None:
        parameters['stop'] = stop
    if max_new_tokens is not None:
        parameters['max_new_tokens'] = max_new_tokens
    if top_k is not None:
        parameters['top_k'] = top_k
    if top_p is not None:
        parameters['top_p'] = top_p
    if no_repeat_ngram_size is not None:
        parameters['no_repeat_ngram_size'] = no_repeat_ngram_size
    if seed is not None:
        parameters['seed'] = seed

    payload = {
        "inputs": prompt,
        "parameters": parameters
    }
    return payload


def generate_text_from_prompt(prompt, kwargs):
    global sm_endpt

    response = sm_endpt.predict(
        prompt, kwargs)
    return response


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
        f"""Summarize the following text in 100 words:\n\n{text}\n\nSummary:""",
        kwargs={
            'temperature': 0.2,  # Low temperature for summarization
            'max_new_tokens': max_new_tokens,
            'seed': seed
        }
    )

def ask(context, question, seed=None, max_new_tokens=200):
    return generate_text_from_prompt(
        f"""CONTEXT:\n{context}\n{question}""",
        kwargs={
            'temperature': 0.05,  # Lowest temperature for accuracy
            'max_new_tokens': max_new_tokens,
            'seed': seed
        }
    )

def extract_question(text, seed=None):
    return generate_text_from_prompt(
        f"""EXTRACT QUESTIONS\nContext:\n{text}\nQuestion:""",
        kwargs={
            'temperature': 1.0,  # Maximum temperature for creativity
            'seed': seed
        }
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



# ----- Classes -----

class SMEndpoint():

    def __init__(self, client):
        self.__client__ = client
        self.__endpoints__ = None
        self.__runtime__ = boto3.client('sagemaker-runtime', region_name=REGION)
        self.__selected_endpoint__ = None

    def refresh_endpoints(self):
        global ENDPOINT_NAME
        try:
            response = self.__client__.list_endpoints()
        except Exception as e:
            return f'Error: {e}'
        endpoints = response['Endpoints']
        self.__endpoints__ = [ endpoint['EndpointName'] for endpoint in endpoints if endpoint['EndpointStatus'] == 'InService' ]
        if len(self.__endpoints__) > 0:
            ENDPOINT_NAME = self.__endpoints__[0]
        else:
            ENDPOINT_NAME = None
        return response

    def get_first_endpoint(self):
        if self.__endpoints__ is None:
            self.refresh_endpoints()
        if len(self.__endpoints__) == 0:
            self.__selected_endpoint__ = 'None'
        else:
            self.__selected_endpoint__ = self.__endpoints__[0]
        return self.__selected_endpoint__

    def select_endpoint(self, endpoint_id):
        if endpoint_id in self.__endpoints__:
            self.__selected_endpoint__ = endpoint_id
            return f'Selected SageMaker endpoint id: {endpoint_id}'
        else:
            return f'Unable to select {endpoint_id} which does not exist'

    def list(self):
        if self.__endpoints__ is None:
            self.refresh_endpoints()
        if len(self.__endpoints__) > 0:
            response = '\n'.join(self.__endpoints__)
        else:
            response = 'There are no SageMaker endpoints'

        print(response)
        return response

    def invoke(self, payload_str):
        payload = json.dumps(payload_str).encode('utf-8')

        try:
            response = self.__runtime__.invoke_endpoint(
                EndpointName=self.__selected_endpoint__,
                ContentType='application/json',
                Body=payload)
        except Exception as e:
            response = f'Error: {e}'
            return response

        response_dict = json.loads(response['Body'].read())
        print(response_dict)

        if self.__selected_endpoint__.startswith('hf-text2text-flan'):
            # Handle Flan endpoint return format
            if 'generated_texts' in response_dict:
                txt_list = response_dict['generated_texts']
                if len(txt_list) > 0:
                    return txt_list[0]
            print("Expected to find a response in the format:\n{'generated_texts': ['<answer>']}")
            print(f'But response = {response}')
            return response

        elif self.__selected_endpoint__.startswith('huggingface-llm-falcon') or \
            self.__selected_endpoint__.startswith('hf-llm-falcon'):
            # Handle Falcon endpoint return format
            if len(response_dict) > 0:
                if 'generated_text' in response_dict[0]:
                    return response_dict[0]['generated_text']
            print("Expected to find a response in the format:\n[{'generated_text': '<answer>'}]")
            print(f'But response = {response}')
            return response

        elif self.__selected_endpoint__.startswith('huggingface-pytorch-tgi'):
            if len(response_dict) > 0 and len(response_dict[0]) > 0:
                if 'generated_text' in response_dict[0][0]:
                    return response_dict[0][0]['generated_text']
            print("Expected to find a response in the format:\n[[{'generated_text': '<answer>'}]]")
            print(f'But response = {response}')
            return response
        else:
            return f'invoke() not implemented for {self.__selected_endpoint__}'
 

    def invoke_text(self, prompt):
        try:
            response = self.__runtime__.invoke_endpoint(
                EndpointName=self.__selected_endpoint__,
                ContentType='application/x-text',
                Body=prompt.encode("utf-8"))
        except Exception as e:
            response = f'Error: {e}'
            return response

        response_dict = json.loads(response['Body'].read())
        if len(response_dict) > 0:
            if 'generated_text' in response_dict[0]:
                return response_dict[0]['generated_text']

        print("Expected to find a response in the format:\n[{'generated_text': '<answer>'}]")
        print(f'But response = {response}')
        return response


    def predict(self, prompt, kwargs):
        if self.__selected_endpoint__ is None:
            return 'ERROR: Cannot generate text because no endpoint has been deployed'

        print(f'Using: {self.__selected_endpoint__}')
        if self.__selected_endpoint__.startswith('hf-text2text-flan'):
            payload = create_payload_from_prompt_flan(prompt, **kwargs)
        elif self.__selected_endpoint__.startswith('huggingface-llm-falcon') or \
            self.__selected_endpoint__.startswith('hf-llm-falcon'):
            payload = create_payload_from_prompt_falcon(prompt, **kwargs)
        elif self.__selected_endpoint__.startswith('huggingface-pytorch-tgi'):
            payload = create_payload_from_prompt_redpajama(prompt, **kwargs)
        else:
            return "Not implemented - for this model, payload function has yet to be created"
        return self.invoke(payload)

    def delete_endpoint(self, endpoint_name):
        if endpoint_name is None:
            return 'No endpoint to delete'
        else:
            return self.__client__.delete_endpoint(EndpointName=endpoint_name)


class LLMModel():
     
    def __init__(self, model, instance_type):
        self.__model_version__ = "*"
        self.__instance_type__ = instance_type
        self.__predictor__ = None
        if model in valid_model_ids:
            self.__model__ = model
        else:  
            raise(f'Invalid model id: {model}')

    def predict(self, payload):
        if self.__predictor__ is not None:
            response = self.__predictor__.predict(payload)
            return response[0]["generated_text"]
        else:
            return f'No model deployed'

    def deploy(self):
        my_model = JumpStartModel(
             model_id = self.__model__,
             instance_type = self.__instance_type__)
        self.__predictor__ = my_model.deploy()

    def delete(self):
        if self.__predictor__ is not None:
            self.__predictor__.delete_model()
            self.__predictor__.delete_endpoint()


class HFModel():
     
    def __init__(self, model_id, hf_task, instance_type):
        self.__hf_task__ = hf_task
        self.__instance_type__ = instance_type
        self.__predictor__ = None
        if model_id in valid_hf_model_ids:
            self.__model_id__ = model_id
        else:  
            raise(f'Invalid model id: {model_id}')

        self.__hub__ = {
            'HF_MODEL_ID': self.__model_id__,
            'HF_TASK': self.__hf_task__
        }
        self.__huggingface_model__ = HuggingFaceModel(
            transformers_version='4.26.0',
            pytorch_version='1.13.1',
            py_version='py39',
            env=self.__hub__,
            role=role,
            image_uri=get_huggingface_llm_image_uri("huggingface", version="0.8.2"),
        )

    def deploy(self):
        self.__predictor__ = self.__huggingface_model__.deploy(
            initial_instance_count=1,
            instance_type=self.__instance_type__,
            container_startup_health_check_timeout=600
        )

    def predict(self, payload):
        if self.__predictor__ is not None:
            return self.__predictor__.predict(payload)
        else:
            return f'No model deployed'

    def delete(self):
        if self.__predictor__ is not None:
            self.__predictor__.delete_model()
            self.__predictor__.delete_endpoint()



sm_endpt = SMEndpoint(client)
ENDPOINT_NAME = sm_endpt.get_first_endpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', default='ml.g5.2xlarge', help='Instance type for deployment')
    parser.add_argument('--deploy_jumpstart',
                        default='',
                        help='Jumpstart model to deploy')
    parser.add_argument('--deploy_hf',
                        default='',
                        help='Hugging Face model to deploy')
    parser.add_argument('--prompt', default='',
                        help='Prompt')
    args = parser.parse_args()

    if args.deploy_jumpstart:
        jsmodel = LLMModel(
             model=args.deploy_jumpstart,
             instance_type=args.instance)
        jsmodel.deploy()

    elif args.deploy_hf:
        hfmodel = HFModel(
            model_id=args.deploy_hf,
            hf_task='text-generation',
            instance_type=args.instance
        )
        predictor = hfmodel.deploy()
    elif args.prompt:
        if ENDPOINT_NAME == 'None':
            print('There are no endpoints to invoke')
        else:
            endpoint_list = sm_endpt.list()
            print(f'Endpoints:\n{endpoint_list}')
            print(f'Selected endpoint: {ENDPOINT_NAME}')
            response = sm_endpt.predict(args.prompt)
            print(f'Response:\n{response}')
