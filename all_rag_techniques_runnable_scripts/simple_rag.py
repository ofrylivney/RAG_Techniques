import os
import sys
import argparse
import time
import pickle
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper_functions import *
# from evaluation.evalute_rag import *

# Add the parent directory to the path since we work with notebooks
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Load environment variables from a .env file (e.g., OpenAI API key)
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Step 1: Define Pydantic schema for structured output
class HapticMaterialProperty(BaseModel):
    material_family: str = Field(description="The superclass of the material in the query, e.g., wood")
    material_class: str = Field(description="The specific type of material, e.g., yellow birch")
    haptic_property: str = Field(description="The haptic property name, e.g., modulus of elasticity")
    value: float = Field(description="The numeric value or range of the haptic property, e.g., 12 or 10-13")
    units: str = Field(description="The units of the haptic property, e.g., MPa")
    source: str = Field(description="The path of the source document or URL that held the information")
    citation: str = Field(description="A sentence or CSV slice that specifies exactly the value that was extracted")

# Step 8: Structured answer model to generate the structured output
def run_structured_answer_model(llm, query: str, retrieved_chunks: List[str], provider='openai') -> HapticMaterialProperty:
    schema = {k: v for k, v in HapticMaterialProperty.schema().items()}
    schema = {"properties": schema["properties"], "required": schema["required"]}
    print(json.dumps(schema, indent=2))
    
    OUTPUT_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

    As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
    the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

    Here is the output schema:

    ```
    {schema}
    ```
    
    Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (```)."""
    json_instruction = OUTPUT_FORMAT_INSTRUCTIONS.format(schema=json.dumps(schema, indent=2))
    print(json_instruction)

    sys_msg_content = f"""
    You are an assistant generating structured output for material properties in the form specified by HapticMaterialProperty model tool from retrieved chunks.":

    {json_instruction}
    """
    print(sys_msg_content)

    sys_msg = SystemMessage(content=sys_msg_content)
                            # "You are an assistant generating structured output for material properties \
                            # in the form specified by HapticMaterialProperty model tool.")
    human_msg = HumanMessage(content=f"Query: {query}\nRetrieved Chunks: {retrieved_chunks[3]}, \
                             \n You must return output based on structured output of {schema}")
    if provider == 'openai':
        structured_llm = llm.with_structured_output(HapticMaterialProperty)
    elif provider == 'ollama':
        structured_llm = llm.with_structured_output(HapticMaterialProperty, include_raw=True)
    else:
        structured_llm = llm.with_structured_output(HapticMaterialProperty)
    result = structured_llm.invoke([sys_msg, human_msg])
    invokation_counter = 0
    while (len(result['raw'].tool_calls) == 0 and invokation_counter<=100):
        invokation_counter = invokation_counter +1
        print(f'number of failed attempts at tool call: {invokation_counter}')
        print(result['raw'].content)
        result = structured_llm.invoke([sys_msg, human_msg])
    # Assuming response contains the necessary fields for structured output
    return HapticMaterialProperty.parse_obj(result)

class SimpleRAG:
    """
    A class to handle the Simple RAG process for document chunking and query retrieval.
    """

    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retrieved=2, embed_model_provider='openai'):
        """
        Initializes the SimpleRAGRetriever by encoding the PDF document and creating the retriever.

        Args:
            path (str): Path to the PDF file to encode.
            chunk_size (int): Size of each text chunk (default: 1000).
            chunk_overlap (int): Overlap between consecutive chunks (default: 200).
            n_retrieved (int): Number of chunks to retrieve for each query (default: 2).
        """
        print("\n--- Initializing Simple RAG Retriever ---")

        # Get the parent directory of the file in the path variable
        parent_dir = os.path.dirname(path)
        # Get the grandparent directory (parent of the parent_dir)
        grandparent_dir = os.path.dirname(parent_dir)
        # Define the embeddings directory as a sibling to the parent directory
        embeddings_dir = os.path.join(grandparent_dir, "embeddings")
        # Ensure the embeddings directory exists
        os.makedirs(embeddings_dir, exist_ok=True)

        # Define paths to save the vector store and metadata in the embeddings directory
        base_filename = os.path.splitext(os.path.basename(path))[0]
        vector_store_path = os.path.join(embeddings_dir, f"{base_filename}_vector_store.faiss")
        metadata_path = os.path.join(embeddings_dir, f"{base_filename}_vector_store_metadata.pkl")

        if embed_model_provider == 'openai':
            embeddings = OpenAIEmbeddings()
        elif embed_model_provider == 'ollama':
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
        else:
            embeddings = OpenAIEmbeddings()

        # Check if the vector store already exists
        if os.path.exists(vector_store_path) and os.path.exists(metadata_path):
            print("Loading vector store from disk...")
            self.vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            # Encode the PDF document into a vector store if not already stored
            print("Processing and encoding the document...")
            start_time = time.time()
            content = document_loader(path)

            # Encode the content into vectors (embedding generation)
            self.vector_store = encode_from_string(content, embeddings=embeddings, chunk_size=chunk_size, chunk_overlap=chunk_overlap) # embeddings, self.metadata
            self.vector_store.metadata = {'vector_store_type': 'FAISS', 
                                          'text_splitter': 'RecursiveCharacterTextSplitter',
                                          'embedding_model': os.getenv('EMBEDDING_MODEL'),
                                          'chunk_size': chunk_size,
                                          'chunk_overlap': chunk_overlap}
            
            # dim = self.vector_store.shape[1]  # Get the dimension of the embeddings
            # # Create a FAISS index
            # self.vector_store = faiss.IndexFlatL2(dim)
            # self.vector_store.add(self.vector_store)

            self.time_records = {'Chunking': time.time() - start_time}
            print(f"Chunking Time: {self.time_records['Chunking']:.2f} seconds")

            # Save the FAISS index and metadata to disk using langchain's save method
            self.vector_store.save_local(vector_store_path)
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.vector_store.metadata, f)
            print(f"Vector store saved to {vector_store_path}")

        # Create a retriever from the vector store
        self.chunks_query_retriever = self.vector_store.as_retriever(search_kwargs={"k": n_retrieved})

    def run(self, query):
        """
        Retrieves and displays the context for the given query.

        Args:
            query (str): The query to retrieve context for.

        Returns:
            tuple: The retrieval time.
        """
        # Measure time for retrieval
        start_time = time.time()
        self.context = retrieve_context_per_question(query, self.chunks_query_retriever)
        # self.time_records['Retrieval'] = time.time() - start_time
        # print(f"Retrieval Time: {self.time_records['Retrieval']:.2f} seconds")

        # Display the retrieved context
        show_context(self.context)
        
# Read Document from path
def document_loader(file_path: str) -> str:
    """Load the document from a file path."""
    with open(file_path, 'r') as file:
        return file.read()

# Function to validate command line inputs
def validate_args(args):
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")
    if args.n_retrieved <= 0:
        raise ValueError("n_retrieved must be a positive integer.")
    return args


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Encode a PDF document and test a simple RAG.")
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to encode.")
    parser.add_argument("--embeddings", type=str, default="openai",
                        help="embedding model provider choice"),
    parser.add_argument("--chat_model", type=str, default="openai",
                        help="Chat model provider choice"),
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Size of each text chunk (default: 1000).")
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="Overlap between consecutive chunks (default: 200).")
    parser.add_argument("--n_retrieved", type=int, default=2,
                        help="Number of chunks to retrieve for each query (default: 2).")
    parser.add_argument("--query", type=str, default="What is the main cause of climate change?",
                        help="Query to test the retriever (default: 'What is the main cause of climate change?').")
    parser.add_argument("--evaluate", type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="Whether to evaluate the retriever's performance (True or False, default: False).")


    # Parse and validate arguments
    return validate_args(parser.parse_args())


# Main function to handle argument parsing and call the SimpleRAGRetriever class
def main(args):
    # Initialize the SimpleRAGRetriever
    simple_rag = SimpleRAG(
        path=args.path,
        embed_model_provider=args.embeddings,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        n_retrieved=args.n_retrieved
    )

    # Retrieve context based on the query
    simple_rag.run(args.query)
    if args.chat_model == 'openai':
        llm = ChatOpenAI(model='gpt-4o-mini')
    else:
        llm = ChatOllama(model='llama3.1:70b', temperature=0.5, format='json')  #'qwen2.5:72b' 'llama3-groq-tool-use:70b'  
    
    structured_output = run_structured_answer_model(llm, args.query, simple_rag.context, provider='ollama')
    print(structured_output)
    # Evaluate the retriever's performance on the query (if requested)
    if args.evaluate:
        evaluate_rag(simple_rag.chunks_query_retriever)


if __name__ == '__main__':
    # Call the main function with parsed arguments
    main(parse_args())
