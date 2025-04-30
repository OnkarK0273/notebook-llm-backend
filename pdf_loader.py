from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from typing import TypedDict,Annotated
from propmt import query_genration_propmt,genration_prompt
from langchain.schema.runnable import RunnableParallel, RunnableLambda


import json
load_dotenv()

# llm's

llm = ChatGoogleGenerativeAI( model="gemini-2.0-flash"  )
embedding_llm = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# pdf loader

loader = PyPDFLoader("pdf/digital-marketing.pdf")
docs = loader.load()

# splitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

split_doc = text_splitter.split_documents(docs)

# embadding > vectorstore

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="pdfs",
#     embedding=embedding_llm
# )

#  vector_store.add_documents(documents=split_doc)


# strucutred output format

class Review(TypedDict):
    query1: Annotated[str, "genrate query to elemented complexity or ambiguous from the user query"]
    query2: Annotated[str, "genrate query to elemented complexity or ambiguous from the user query"]
    query3: Annotated[str, "genrate query to elemented complexity or ambiguous from the user query"]

strutured_model = llm.with_structured_output(Review)

# fanout reterival method--------------

# 1. query genration

query = input("query > ")

propmt = query_genration_propmt.invoke({"user_query":query})

res = strutured_model.invoke(propmt)

query1 = res.get("query1")
query2 = res.get("query2")
query3 = res.get("query3")

# 2. parallel reterival

def reterival_doc(query ,embedder = embedding_llm):

    retrivel = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="pdfs",
    embedding=embedder
    )

    retrival_result = retrivel.similarity_search(
        query=query
    )

    return retrival_result

parallel_retrieval = RunnableParallel({
    'retrieval1': RunnableLambda(lambda _: reterival_doc(query1)),
    'retrieval2': RunnableLambda(lambda _: reterival_doc(query2)),
    'retrieval3': RunnableLambda(lambda _: reterival_doc(query3))
})

results = parallel_retrieval.invoke({})


# context mearging (filterout unique documents)-------------------

list = []

for key, value_list in results.items():
    for item in value_list:
        list.append(item.page_content)

unique_doc = set(list)

context = ""

for content in unique_doc:
    context += content

print(context)

# answer genration--------------------------------

propmt2 = genration_prompt.invoke({'context':context,'query':query})

final_res = llm.invoke(propmt2)

print('----- final response -----')
print(final_res.content)