import os
from dotenv import load_dotenv
from langchain.llms import HuggingFaceEndpoint 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import gradio as gr

# Load environment variables from .env file
load_dotenv()

# Set constants for Hugging Face model and API token and the model id
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
TOKEN_API = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Check if the API token is loaded correctly
if not TOKEN_API:
    raise ValueError("Hugging Face API token not found. Ensure that the .env file contains the 'HUGGINGFACEHUB_API_TOKEN'.")

# Initialize the HuggingFaceEndpoint with model configuration
try:
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        max_length=200,
        temperature=0.6,
        huggingfacehub_api_token=TOKEN_API  # Fix token key typo here
    )
    
except Exception as e:
    raise RuntimeError(f"Failed to initialize HuggingFaceEndpoint: {e}")

def chatbot(complaint: str) -> str:
    # Define a PromptTemplate to format the prompt with user input
    prompt = PromptTemplate(input_variables=["complaint"], 
                            template="I am a customer service representative. I received the following complaint: {complaint}. My response is:"
    )
    
    # Create a language model chain with the defined prompt template
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        response = chain.run(complaint)
        return response
    except Exception as e:
        return f"Error generating response: {e}"
    
demo = gr.Interface(fn=chatbot, inputs="text", outputs="text")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
