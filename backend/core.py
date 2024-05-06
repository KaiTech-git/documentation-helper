import os

from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.hub import pull
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

# from langchain.chains.retrieval_qa.base import RetrievalQA

from typing import Any, List, Tuple
from langchain_pinecone import PineconeVectorStore


def run_llm(query: str, chat_history:List[Tuple[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        embedding=embeddings, index_name=os.environ["INDEX_NAME"]
    )
    chat = OpenAI(verbose=True, temperature=0)
    
    # qa = RetrievalQA.from_chain_type(
    #     llm=chat,
    #     chain_type="stuff",
    #     retriever=docsearch.as_retriever(),
    #     return_source_documents=True,
    # )
    
    #New version but the implementation but I'm not sure how to use it wit memory so I change the implementation to ConversationalRetrievalChain
    
    # retrieval_qa_chat_prompt = pull("langchain-ai/retrieval-qa-chat")
    # combine_docs_chain = create_stuff_documents_chain(
    #     llm=chat, prompt=retrieval_qa_chat_prompt
    # )
    # retrieval_chain = create_retrieval_chain(
    #     retriever=docsearch.as_retriever(), combine_docs_chain=combine_docs_chain
    # )

    # return retrieval_chain.invoke({"input": query, "chat_history": chat_history})
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )

    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="What is LangChain?"))

if __name__ == "__main__":
    res = run_llm(
        "What is create_retrieval_chain?"
    )
    print(res["answer"])
