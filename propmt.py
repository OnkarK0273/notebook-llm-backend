from langchain_core.prompts import PromptTemplate,ChatPromptTemplate

system_propmt = """
your are intelligent ai assistant that helpfull for to genrate 3 query to elemented complexity or ambiguous from the user query

rules:
follow strict json output format

example:
user_query - How to fix Python error?

output - {{
    "query1" :"What are common types of errors in Python and how to fix them?",
    "query2" :"How can I debug a syntax error in Python code?",
    "query3" :"How do I resolve import errors in Python projects?",
}}
"""

query_genration_propmt = ChatPromptTemplate([
    ("system",system_propmt),
    ("human","user_query - {user_query}")
])


genration_prompt = ChatPromptTemplate([
    ('system', 'You are an assistant helping to answer user queries based on the following content. If the answer to the query is not found in the content, respond with: "This query is not present in the document." Here is the content:\n\n{context}'),
    ('human', '{query}')
])