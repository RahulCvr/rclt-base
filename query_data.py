import argparse
import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB with Azure OpenAI embeddings
    embedding_function = AzureOpenAIEmbeddings(
        azure_deployment=os.environ['AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME'],  # Replace with your deployment name
        openai_api_version="2024-12-01-preview",
        azure_endpoint=os.environ['AZURE_OPENAI_EMBEDDING_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_EMBEDDING_API_KEY']
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Use Azure OpenAI chat model
    model = AzureChatOpenAI(
        azure_deployment=os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME'],  # Replace with your deployment name
        openai_api_version="2024-12-01-preview",
        azure_endpoint=os.environ['AZURE_OPENAI_CHAT_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_CHAT_API_KEY'],
    )
    response = model.invoke(prompt)
    response_text = response.content

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
