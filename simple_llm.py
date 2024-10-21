import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceEndpoint
import gradio as gr

# Load environment variables from .env file
load_dotenv()

# Set constants for Hugging Face model and API token and the model id
#MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_ID='meta-llama/Meta-Llama-3-8B-Instruct'
API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Check if the API token is loaded correctly
if not API_TOKEN:
    raise ValueError("Hugging Face API token not found. Ensure that the .env file contains the 'HUGGINGFACEHUB_API_TOKEN'.")

# Initialize the HuggingFaceEndpoint with model configuration
try:
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        max_length=200,
        temperature=0.6,
        huggingfacehub_api_token=API_TOKEN
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize HuggingFaceEndpoint: {e}")

# Define the chatbot function to interact with the model
def chatbot(prompt: str) -> str:
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"Error generating response: {e}"

# Define the Gradio interface for the chatbot
demo = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="LLM-Powered Chatbot",
    description="Ask anything and get responses powered by the Meta-Llama-3-8B model."
)

# Launch the Gradio interface on the specified port and server name
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
