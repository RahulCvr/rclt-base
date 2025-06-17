# Langchain RAG Tutorial - Azure OpenAI Version

## Install dependencies

1. Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing `onnxruntime` through `pip install onnxruntime`. 

    - For MacOS users, a workaround is to first install `onnxruntime` dependency for `chromadb` using:

    ```python
     conda install onnxruntime -c conda-forge
    ```
    See this [thread](https://github.com/microsoft/onnxruntime/issues/11037) for additonal help if needed. 

     - For Windows users, follow the guide [here](https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file) to install the Microsoft C++ Build Tools. Be sure to follow through to the last step to set the enviroment variable path.


2. Now run this command to install dependenies in the `requirements.txt` file. 

```python
pip install -r requirements.txt
```

3. Install markdown depenendies with: 

```python
pip install "unstructured[md]"
```

## Azure OpenAI Setup

1. Create a `.env` file in the project root with your Azure OpenAI credentials:

```
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
```

2. Update the deployment names in the code files to match your Azure OpenAI deployments:
   - Replace `"text-embedding-ada-002"` with your embedding model deployment name
   - Replace `"gpt-4"` with your chat model deployment name

3. The endpoint URL is configured as `https://cline-azure-openai-west.openai.azure.com/` - update this in all files if your endpoint is different.

## Create database

Create the Chroma DB using Azure OpenAI embeddings.

```python
python create_database.py
```

## Query the database

Query the Chroma DB using Azure OpenAI.

```python
python query_data.py "How does Alice meet the Mad Hatter?"
```

## Compare embeddings

Test Azure OpenAI embeddings comparison.

```python
python compare_embeddings.py
```

> You'll need an Azure OpenAI account and resource set up with deployed models for embeddings and chat completions.

Here is a step-by-step tutorial video: [RAG+Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami).
