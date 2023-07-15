#!/usr/bin/env python

# Run with:
# streamlit run ui_streamlit.py

# On EC2:
# ssh -L 8080:localhost:8080 ec2-user@hostname
# streamlit run ui_streamlit.py --server.runOnSave true --server.port=8080
# (Browser to http://127.0.0.1:8080/)
#
# Run with screen in detached mode using
# screen -S llm -d -m streamlit run ui_streamlit.py --server.runOnSave true --server.port 8080

import base64
import boto3
import os
import re
import sagemaker
import streamlit as st

from llm_deploy import sm_endpt
from sagemaker.session import Session
from sagemaker.jumpstart.model import JumpStartModel
from streamlit_chat import message
from dotenv import load_dotenv


load_dotenv()

DEPLOY_KEY = os.getenv('DEPLOY_KEY')
STREAMLIT_PASSWORD = os.getenv('STREAMLIT_PASSWORD')
MAX_ENDPOINTS = 2
MAX_G52XL_ENDPOINTS = 2
MAX_G512XL_ENDPOINTS = 1



# ----- SageMaker Setup -----
sagemaker_session = Session()
aws_role = sagemaker_session.get_caller_identity_arn()
aws_region = boto3.Session().region_name
sess = sagemaker.Session()

print(f'Using role {aws_role} in region {aws_region}. boto3=={boto3.__version__}. sagemaker=={sagemaker.__version__}')

if not os.path.exists('download_dir'):
    os.mkdir('download_dir')


# ----- LLM Subroutines -----
from llm_deploy import generate_text_from_prompt, extract_paragraphs_from_html, \
    download_url_text, extract_pages, ask, summarize, create_qna_pairs_from_paras

def extract_context_from_url(url, first_para=1, last_para=10):
    paragraph_list = extract_paragraphs_from_html(
        download_url_text(url)
    )[first_para:last_para+1]
    context = '\n\n'.join(paragraph_list)
    return context, paragraph_list

def extract_context_from_file(pdf_file):
    pages = extract_pages(pdf_file, max_pages=12)
    pages_txt = '\n\n'.join(pages[0:3] + pages[-3:])
    return pages_txt

def get_keywords_from_context(context):
    key_words = generate_text_from_prompt(
        f'FIND KEY WORDS\n\nContext:\n{context}\nKey Words:',
        seed=12345
    )
    return key_words

def get_point_summary(para_list, n=5, max_new_tokens=100, seed=None):
    summary_list = []
    for i, x in enumerate(para_list[:n]):
        if i == 0 and seed is not None:
            summary_txt = summarize(x[:1500], max_new_tokens=max_new_tokens, seed=seed)
        else:
            summary_txt = summarize(x[:1500], max_new_tokens=max_new_tokens)
        summary_list.append(summary_txt)
    return summary_list



# ----- Comprehend -----

comprehend = boto3.client('comprehend')

def get_key_phrases(txt_list):
    if len(txt_list) > 25:
        print("Maximum is 25 paragraphs")
        return []
    try:
        response = comprehend.batch_detect_key_phrases(
            TextList=txt_list,
            LanguageCode='en'
        )
    except comprehend.exceptions.BatchSizeLimitExceededException as e:
        print(f'{e}')
        print(f'Your batch size is: {len(txt_list)}')
    except comprehend.exceptions.TextSizeLimitExceededException as e:
        print(f'{e}')
        print(f'Your maximum input text size is: {max([ len(x) for x in txt_list ])}')
    except Exception as e:
        print(f'Error {e}')
        print('If this is an AccessDeniedException, you might need to add `BatchDetectKeyPhrases` permissions to {aws_role}')
    else:
        print(response)
        results = response['ResultList'][0]
        if len(results) == 0:
            print('No key phrases found')
            return [], ''

        phrase_list = set()
        top_phrase = ''

        if 'KeyPhrases' in results:
            key_phrases = results['KeyPhrases']
            # print(f'Key Phrases: {key_phrases}')
            if 'Score' in key_phrases[0]:
                prob_max = float(key_phrases[0]['Score'])
                top_phrase = key_phrases[0]['Text']
                for p in key_phrases:
                    print(f'Key Phrase: {p}')
                    prob_score = float(p['Score'])
                    if prob_score > prob_max:
                        top_phrase = p['Text']
                    if prob_score > 0.995:
                        phrase_list.add(p['Text'])
                return list(phrase_list), top_phrase
        else:
            print(f'Results dictionary in unrecognized format: {results}')
    return [], ''



# ----- Streamlit  -----

st.set_page_config(page_title='🎓 GenAI Demo for Education 🎓')


response = sm_endpt.refresh_endpoints()
print(f'Refreshing endpoints:\n{response}')
if len (sm_endpt.__endpoints__) > 0:
    endpoint_id = sm_endpt.__endpoints__[0]
    response = sm_endpt.select_endpoint(endpoint_id)
    print(f'Selected endpoint: {endpoint_id}')
    print(response)


# ----- Streamlit Password Checker-----

def ui_password_verifier(skip_password=False):

    if skip_password:
        return True

    def password_entered():
        if st.session_state["password"] == STREAMLIT_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Password",
            type="password",
            on_change=password_entered,
            key="password"
        )
        return False

    elif not st.session_state["password_correct"]:
        st.text_input(
            "Password",
            type="password",
            on_change=password_entered,
            key="password"
        )
        st.error("ℹ️ Password is incorrect")
        return False

    else:
        return True



# ----- Streamlit Main -----
if ui_password_verifier(skip_password=False):
    st.title('🎓 GenAI Demo for Education 🎓')

    sm_endpt.refresh_endpoints()
    point_summary = None

    tab_model, tab_context, tab_chat, tab_paper, tab_summary, tab_qa, tab_marking = st.tabs([
        "Model",
        "Data Source",
        "Chat",
        "Paper",
        "Summarize",
        "Generate Q&A",
        "Marking"
        ])

    # ..... [Tab] Model .....
    with tab_model:
        if len(sm_endpt.__endpoints__) > 0:
            for endpt in sm_endpt.__endpoints__:
                st.success(endpt, icon="✅")
        else:
            st.info('No endpoints are active', icon='ℹ️')

        with st.form('selectendpointform', clear_on_submit=False):
            endpoint_id = st.selectbox(
                'Choose an endpoint',
                list(sm_endpt.__endpoints__))

            col_select_endpt, col_refresh_endpts, col_delete_endpt = st.columns([1, 1, 1])
            with col_select_endpt:
                submit_select = st.form_submit_button('Use this endpoint', disabled=len(sm_endpt.__endpoints__) == 0)
            with col_refresh_endpts:
                submit_endpointcheck = st.form_submit_button('Refresh endpoints')
            with col_delete_endpt:
                submit_delete = st.form_submit_button('Delete this endpoint', disabled=len(sm_endpt.__endpoints__) == 0)

            if submit_select:
                response = sm_endpt.select_endpoint(endpoint_id)
                st.info(response)

            if submit_endpointcheck:
                try:
                    response = sm_endpt.refresh_endpoints()
                    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                        st.info('Endpoints refreshed')
                    else:
                        st.error('Failed to refresh endpoints\nResponse:\n{response}')
                except Exception as e:
                    st.error(f'Error: {e}')

            if submit_delete:
                with st.spinner(f'Deleting {endpoint_id}...'):
                    try:
                        response = sm_endpt.delete_endpoint(endpoint_id)
                        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                            st.info('Endpoint deleted')
                        else:
                            st.error('Failed to delete endpoint\nResponse:\n{response}')
                    except Exception as e:
                        response = f'Error: {e}'
                    st.info(response)

                    # Refresh endpoints after endpoint is deleted
                    response = sm_endpt.refresh_endpoints()


        with st.form('deployform', clear_on_submit=True):
            instance_type = st.selectbox(
                'Select an instance type',
                ('ml.g5.2xlarge', 'ml.g5.12xlarge'))

            model_id = st.selectbox(
                'Select a model to deploy',
                ('huggingface-text2text-flan-t5-xl',
                 'huggingface-text2text-flan-t5-xxl',
                 'huggingface-llm-falcon-7b-bf16',
                 'huggingface-llm-falcon-7b-instruct-bf16',
                 'huggingface-llm-falcon-40b-bf16',
                 'huggingface-llm-falcon-40b-instruct-bf16',
                 'huggingface-textgeneration1-redpajama-incite-instruct-7B-v1-fp16'
                ))

            deploy_key = st.text_input('Deploy Token', type='password')
            submit_deploy = st.form_submit_button('Deploy JumpStart')
            if submit_deploy:
                if deploy_key == DEPLOY_KEY:
                    response = sm_endpt.refresh_endpoints()
                    if len(sm_endpt.__endpoints__) >= MAX_ENDPOINTS:
                        st.error('Deployment failed. Maximum number of endpoints has been reached')
                    else:
                        with st.spinner(f'Deploying JumpStart {model_id}...'):
                            try:
                                my_model = JumpStartModel(
                                    model_id=model_id,
                                    instance_type=instance_type)
                                predictor = my_model.deploy()
                                response = 'JumpStart Model deployed'
                            except Exception as e:
                                response = f'Error: {e}'
                        st.info(response)
                        response = sm_endpt.refresh_endpoints()
                        st.info(response)
                        response = sm_endpt.select_endpoint(model_id)
                        st.info(response)
                else:
                    st.error('Please enter a valid deploy key')



    # ..... [Tab] Data Source .....
    with tab_context:
        url = st.text_input('Enter a URL', value='https://en.wikipedia.org/wiki/Quantum_computing')
        if 'summary_list' in st.session_state:
            del st.session_state['summary_list']

        pdf_files = [ x for x in os.listdir('download_dir') if x.endswith('.pdf') ]
        pdf_file = st.selectbox('Or select a PDF file', tuple(pdf_files))

        uploaded_file = st.file_uploader('Or upload a file', type='pdf', accept_multiple_files=False)
        if uploaded_file is not None:
            with open(f'download_dir/{uploaded_file.name}', 'wb') as f:
                bytes = uploaded_file.getvalue()
                f.write(bytes)
            st.write(f'Uploaded file: {uploaded_file.name}')
            try:
                pdf_data = base64.b64encode(bytes).decode('utf-8')
                st.markdown(
                    F'<iframe src="data:application/pdf;base64,{pdf_data}" width="700" height="400" type="application/pdf"></iframe>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                print(f'{e}')


    # ..... [Tab] Chat .....
    with tab_chat:
        result = []
        message_history = []
        placeholder = st.empty()
        with st.form('myform', clear_on_submit=False):
            data_source = st.selectbox('Data Source', (url, pdf_file))
            query_text = st.text_input('Enter your question:', value='What is quantum computing?')
            submitted = st.form_submit_button('Submit')
            if submitted:
                message(query_text, is_user=True)
                with st.spinner('Calculating...'):
                    if data_source == pdf_file:
                        context = extract_context_from_file(f'download_dir/{pdf_file}')
                        response = ask(context[:2000], query_text)
                    else:
                        context, _ = extract_context_from_url(url, first_para=1, last_para=7)
                        response = ask(context[:2000], query_text)
                    result.append(response)
            if len(result):
                message(response)

    # ..... [Tab] Paper .....
    with tab_paper:
        st.info(f'Data Source: {pdf_file}')
        with st.form('paperform', clear_on_submit=True):
            pdf_pages = extract_pages(f'download_dir/{pdf_file}', max_pages=12)
            context = '\n\n'.join(pdf_pages[1:3] + pdf_pages[-3:])
            print(context)

            submit_gist = st.form_submit_button('What is the main gist of the paper?')
            if submit_gist:
                with st.spinner('Calculating...'):
                    response = ask(context[:2000], 'What is the main gist of the paper?', max_new_tokens=200)
                    st.info(response)
            submit_prob = st.form_submit_button('What is the problem being solved?')
            if submit_prob:
                with st.spinner('Calculating...'):
                    response = ask(context[:2000], 'What is the problem being solved?', max_new_tokens=200)
                    st.info(response)
            submit_concl = st.form_submit_button('What is the conclusion of the paper?')
            if submit_concl:
                with st.spinner('Calculating...'):
                    response = ask(context[:2000], 'What is the conclusion of the paper?', max_new_tokens=200)
                    st.info(response)

    # ..... [Tab] Summary .....
    with tab_summary:
        st.info(f'Data Source: {url}')
        with st.form('summarypointsform', clear_on_submit=False):
            col_point_by_point_summary, col_rand_seed = st.columns([1, 1])
            with col_point_by_point_summary:
                submit_summary_n = st.form_submit_button('Point-by-point summary')

            if submit_summary_n:
                with st.spinner('Generating point-by-point summary...'):
                    if data_source == pdf_file:
                        st.info('Point-by-point summary for pdf files not yet implmented')
                    else:
                        _, para_list = extract_context_from_url(url)
                        summary_list = get_point_summary(para_list, n=5)
                        st.session_state.summary_list = summary_list

                        response_list = []
                        for i, x in enumerate(summary_list):
                            response_txt = f'{i+1}. {x}'
                            response_list.append(response_txt)
                        st.info('\n'.join(response_list))

        with st.form('summaryquizform', clear_on_submit=False):
            submit_summary_q = st.form_submit_button('Create fill in the blanks quiz')

            if submit_summary_q:
                with st.spinner('Removing key words using Amazon Comprehend...'):
                    if data_source == pdf_file:
                        st.info('Keyword removal for pdf files not yet implmented')
                    else:
                        if 'summary_list' not in st.session_state:
                            _, para_list = extract_context_from_url(url)
                            summary_list = get_point_summary(para_list, n=5)
                            st.session_state.summary_list = summary_list
                        response_list = []
                        for i, txt in enumerate(st.session_state.summary_list):
                            print(txt)
                            key_phrases, key_phrase = get_key_phrases([txt])
                            if len(key_phrases) > 0:
                                if key_phrase in txt:
                                    new_txt = re.sub(
                                        pattern=key_phrase,
                                        repl='_____', string=txt
                                        )
                                    response_list.append(
                                        f'{+1}. {new_txt} (**Ans**: {key_phrase})'
                                    )
                            else:
                                print(f'No key phrases found: key_phrases={key_phrases}. key_phrase={key_phrase}')
                                response_list.append(
                                    f'{+1}. {txt}'
                                )
                        st.markdown('\n'.join(response_list))

        with st.form('summaryform', clear_on_submit=False):
            submit_summary = st.form_submit_button('Summarize')
            if submit_summary:
                with st.spinner('Summarizing...'):
                    if data_source == pdf_file:
                        st.info('Final summarization for pdf files not yet implmented')
                    else:
                        if 'summary_list' not in st.session_state:
                            _, para_list = extract_context_from_url(url)
                            summary_list = get_point_summary(para_list, n=5)
                            st.session_state.summary_list = summary_list
                        response = summarize('\n\n'.join(st.session_state.summary_list))
                        key_words, _ = get_key_phrases(st.session_state.summary_list)
                        key_words = ', '.join(key_words)
                        st.info(f"""{response}\n\nKey phrases: {key_words}""")

    # ..... [Tab] Q&A .....
    with tab_qa:
        st.info(f'Data Source: {url}')
        with st.form('qaform', clear_on_submit=True):
            submit_qa = st.form_submit_button('Generate Q&A Pairs')

            if submit_qa:
                with st.spinner('Generation Q&A pairs...'):
                    if data_source == pdf_file:
                        st.info('Q&A pair generation for pdf files not yet implmented')
                    else:
                        _, para_list = extract_context_from_url(url)
                        qn_list, ans_list = create_qna_pairs_from_paras(para_list[:5])

                        response_list = []
                        for i in range(len(qn_list)):
                            response_txt = f'{i+1}. **Question**: {qn_list[i]} **Answer**: {ans_list[i]}'
                            response_list.append(response_txt)
                            # st.info(response_txt)
                        st.markdown('\n'.join(response_list))

    # ..... [Tab] Marking .....
    with tab_marking:
        marking_url = 'https://en.wikipedia.org/wiki/Quantum_computing'
        st.info(f'Data Source: {marking_url}')
        with st.form('markingform', clear_on_submit=False):
            st.markdown('### Quiz')
            st.info('What is quantum computing? (2 marks)')
            answer_text = st.text_input('Enter your answer:', value='Quantum computing involves using computers that make use of quantum mechanics')
            submit_marking = st.form_submit_button('Check this answer')

            submit_report = False
            if submit_marking:
                with st.spinner('Calculating...'):
                    context, _ = extract_context_from_url(marking_url, first_para=1, last_para=7)

                    # Grading
                    query_list = [
                        'Does this answer mention the use of quantum phenomena?',
                        'Does this answer quantum computing speedups over classical computing?'
                    ]
                    marks = 0
                    for query_text in query_list:
                        prompt = f"""Answer: {answer_text}
                        Question: {query_text}"""
                        response = generate_text_from_prompt(prompt, kwargs={'temperature': 0.05, 'max_new_tokens': 5})
                        if response.lower() == 'yes':
                            st.success(f'✅  {query_text} {response} (1 mark)')
                            marks += 1
                        elif response.lower() == 'no':
                            st.error(f'❌  {query_text} {response} (0 marks)')
                        else:
                            st.error(f'❌  {query_text} no (0 marks)')
                            # st.info(f'ℹ️ {query_text} {response}')
                    
                    st.info(f'ℹ️ Total marks: {marks}')

            submit_report = st.form_submit_button('Report Issue')
            if submit_report:
                st.info('ℹ️ Thank you for reporting. This will be sent for review')
