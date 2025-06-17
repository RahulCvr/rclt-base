import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureOpenAIEmbeddings
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

def main():
    # Get embedding for a word using Azure OpenAI
    embedding_function = AzureOpenAIEmbeddings(
        azure_deployment=os.environ['AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME'],  # Replace with your deployment name
        api_version="2024-12-01-preview",
        azure_endpoint=os.environ['AZURE_OPENAI_EMBEDDING_ENDPOINT'],
        openai_api_key=os.environ['AZURE_OPENAI_EMBEDDING_API_KEY']
    )
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector[:5]}")
    print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    evaluator = load_evaluator("pairwise_embedding_distance", embeddings=embedding_function)
    words = ("apple", "iphone")
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    main()
