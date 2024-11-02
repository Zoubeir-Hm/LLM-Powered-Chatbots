from dotenv import load_dotenv
import os
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
import gradio as gr

# Load environment variables from .env file
load_dotenv()

# Set constants for Hugging Face model and API token and the model id
MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")

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


def chatbot(user_input)->str:

    # Define a PromptTemplate to format the prompt with user input
    prompt = PromptTemplate(
        input_variables=['question'],
        template="""Question : {question}
        please provide step by step Answer : 
        """
    )
    formated_prompt=prompt.format(question=str(user_input))
    try:
        response = llm.invoke(formated_prompt)
        return response
    except Exception as e:
        raise f"Error generating response: {e}"
    
# Define the Gradio interface output
output = gr.Textbox(label="Step by Step")

demo = gr.Interface(fn=chatbot, inputs="text", outputs=output)

# Launch the Gradio interface on the specified port and server name
if __name__== "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

