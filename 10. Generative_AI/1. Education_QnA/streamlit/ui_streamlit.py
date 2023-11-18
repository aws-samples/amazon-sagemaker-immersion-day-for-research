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
import json
import os
import re
import sagemaker
import streamlit as st
import time

from sagemaker.session import Session
from sagemaker.jumpstart.model import JumpStartModel
from streamlit_chat import message


# ----- Setup -----
if "messages" not in st.session_state:
    st.session_state.messages = []

if not os.path.exists('download_dir'):
    os.mkdir('download_dir')

def read_json_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)


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

def get_point_summary(para_list, n=5, max_new_tokens=100):
    summary_list = []
    for i, x in enumerate(para_list[:n]):
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



# ----- Streamlit Main -----
def streamlit_main():
    st.title('🎓 GenAI Demo for Education 🎓')

    point_summary = None

    tab_context, tab_chat, tab_paper, tab_summary, tab_qa, tab_marking = st.tabs([
        "Data Source",
        "Chat",
        "Paper",
        "Summarize",
        "Generate Q&A",
        "Marking"
        ])

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
            data_source = st.selectbox(
                'Data Source',
                (url, pdf_file) if pdf_file else (url,)
            )
            query_text = st.text_input('Enter your question:', value='What is quantum computing?')

            col_submit_question, col_clear_chat = st.columns([1, 1])
            with col_submit_question:
                submitted = st.form_submit_button('Submit')
            with col_clear_chat:
                submit_clear_chat = st.form_submit_button('Clear chat')

            if submit_clear_chat:
                st.session_state.messages = []

            if 'message' in st.session_state:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            if submitted:

                st.chat_message("user").markdown(query_text)
                x = {
                        "role": "user",
                        "content": query_text
                    }
                if 'messages' in st.session_state:
                    st.session_state.messages.append(x)
                else:
                    st.session_state.messages = [x]

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    if data_source == pdf_file:
                        context = extract_context_from_file(f'download_dir/{pdf_file}')
                        response = ask(context[:2000], query_text)
                    else:
                        context, _ = extract_context_from_url(url, first_para=1, last_para=7)
                        response = ask(context[:2000], query_text)
                    result.append(response)
    
                    full_response = ''
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
    
                    message_placeholder.markdown(full_response)


    # ..... [Tab] Paper .....
    with tab_paper:
        if pdf_file:
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
        else:
            st.info('No PDF file has been uploaded')

    # ..... [Tab] Summary .....
    with tab_summary:
        st.info(f'Data Source: {url}')
        with st.form('summarypointsform', clear_on_submit=False):
            col_point_by_point_summary, col_rand_seed = st.columns([1, 1])
            with col_point_by_point_summary:
                submit_summary_n = st.form_submit_button('Point-by-point summary')

            if submit_summary_n:
                with st.spinner('Generating point-by-point summary...'):
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
        with st.form('markingform', clear_on_submit=False):
            st.markdown('### Practice Quiz')

            quiz_file = 'practice_quiz.json'
            quiz = read_json_file(f'download_dir/{quiz_file}')

            answer_list = []
            for i, q in enumerate(quiz):
                question = q['Question']
                evaluate = q['Evaluate']
                sample_answer = q['Sample answer']
                points = sum([e['Marks'] for e in evaluate])
                st.markdown(f'**{i+1}. {question}** ({points} points)')
    
                if len(quiz) == 1:
                    answer_text = st.text_area('Enter your answer:', value=sample_answer, height=20)
                else:
                    answer_text = st.text_input('Enter your answer:', value=sample_answer)
                answer_list.append(answer_text)

            submit_marking = st.form_submit_button(
                'Submit for evaluation')

            final_points = 0
            maximum_points = 0
            if submit_marking:
                with st.spinner('Calculating...'):
                    for i, q in enumerate(quiz):
                        question = q['Question']
                        evaluate = q['Evaluate']
    
                        answer = answer_list[i]
                        st.markdown(f"**{i+1}. Question**: {question}\n**Your Answer**: {answer}")
    
                        # Grading
                        points = 0
                        for j, e in enumerate(evaluate):
                            criteria = e['Criteria']
                            hint_text = e['Hint']
                            m = int(e['Marks'])
                            maximum_points += m
    
                            prompt = """Given the following statment <statement>{answer}</statement> state whether it meets the following criteria <criteria>{criteria}</criteria>. Answer only with yes or no""".format(answer=answer, criteria=criteria)
                            response = generate_text_from_prompt(
                                prompt, temperature=0.04, max_tokens_to_sample=2
                            )
    
                            response = response.strip().lower()
                            if response.startswith('answer: '):
                                response = response.replace('answer: ', '')
    
                            if response == 'yes':
                                unit = 'point' if m == 1 else 'points'
                                st.success(f'✅  {criteria} {response} ({m} {unit})')
                                points += m
                            elif response == 'no':
                                st.error(f'❌  {criteria} {response} (0 points)')
                            else:
                                st.error(f'❌  {criteria} {response} (0 points)')
                                # st.info(f'ℹ️ {criteria} {response}')
                        
                        st.info(f'ℹ️ Total points: {points}')
                        final_points += points

                st.info(f'Final points: {final_points} out of a maximum of {maximum_points}')
     

if __name__ == '__main__':
    streamlit_main()
