# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
# API 키 정보 로드
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
# logging.langsmith("CH12-RAG")


loader = PyPDFDirectoryLoader("./Data/")
docs = loader.load()

# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

# 단계 3: 임베딩(Embedding) 생성
embeddings = OpenAIEmbeddings()

# 단계 4: DB 생성(Create DB) 및 저장
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# 단계 5: 검색기(Retriever) 생성
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

# 단계 6-2: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. and You are also a stock analyst with a positive outlook.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Question: 
{question} 
#Context: 
{context} 

#Answer:"""
)

# 단계 6-2: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt2 = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. and You are also a stock analyst with a negative outlook. You need to rebut the given question.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Question: 
{question} 
#Context: 
{context} 

#Answer:"""
)

prompt3 = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. and You are also an expert in summarization.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Question: 
{question} 
#Context: 
{context} 

#Answer:"""
)

# 단계 7: 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.
pos_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
neg_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
sum_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# 단계 8: 체인(Chain) 생성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | pos_llm
    | StrOutputParser()
)

# 단계 8: 체인(Chain) 생성
chain2 = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt2
    | neg_llm
    | StrOutputParser()
)

chain3 = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt3
    | sum_llm
    | StrOutputParser()
)

# 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
# question = "삼성전자 주식 사고싶은데 어떻게 생각해??"
print('질문을 입력해주세요! input : ')

question = input()
pos_response = chain.invoke(question)
print('긍정:', pos_response)

print('Turn Change')

# 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
neg_response = chain2.invoke(pos_response+'여기에 대해 부정적인 입장에서 반박해줘')
print('부정:', neg_response)

print('Turn Change')

pos_response2 = chain.invoke(neg_response+'여기에 대해 긍정적인 입장에서 반박해줘')
print('긍정반박:', pos_response2)

print('Turn Change')

neg_response2 = chain2.invoke(pos_response2+'여기에 대해 부정적인 입장에서 반박해줘')
print('부정반박:', neg_response2)


sum_question = f"""토론은 다음과 같아 
긍정: {pos_response}
부정: {neg_response}
긍정: {pos_response2}
부정: {neg_response2}
토론 내용을 한줄 요약해줘"""

sum_response = chain.invoke(sum_question)
print('요약:', sum_response)

a1 = pos_response
a2 = neg_response
a3 = pos_response2
a4 = neg_response2
a5 = sum_response

# 텍스트 파일로 저장하는 함수
def save_to_text_file(a1, a2, a3, a4, a5, filename="output.txt"):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(f"긍정: {a1}\n")
        file.write('turn\n')
        file.write(f"부정: {a2}\n")
        file.write('turn\n')
        file.write(f"긍정반박: {a3}\n")
        file.write('turn\n')
        file.write(f"부정반박: {a4}\n")
        file.write('turn\n')
        file.write(f"요약: {a5}\n")

# 텍스트 파일 저장 호출
save_to_text_file(a1, a2, a3, a4, a5, filename="토론_요약.txt")   
    

