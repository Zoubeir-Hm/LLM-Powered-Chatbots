import gradio as gr
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
import os


# Load environment variables from .env file
load_dotenv()

# Set constants for Hugging Face model and API token and the model id
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


# Define a PromptTemplate to format the prompt with user input
prompt = PromptTemplate(
    input_variables=["position", "company", "skills"],
    template="Dear Hiring Manager,\n\nI am writing to apply for the {position} position at {company}. I have experience in {skills}.\n\nThank you for considering my application.\n\nSincerely,\n[Your Name]",
)

# Define a function to generate a cover letter using the llm and user input
def generate_cover_letter(position: str, company: str, skills: str) -> str:
    formatted_prompt = prompt.format(position=position, company=company, skills=skills)
    try:
        response = llm.invoke(formatted_prompt)
        return response
    except Exception as e:
        return f"Error generating response: {e}"

# Define the Gradio interface inputs
inputs = [
    gr.Textbox(label="Position"),
    gr.Textbox(label="Company"),
    gr.Textbox(label="Skills")
]
# Define the Gradio interface output
output = gr.Textbox(label="Cover Letter")

demo = gr.Interface(fn=generate_cover_letter, inputs=inputs, outputs=output)


# Launch the Gradio interface on the specified port and server name
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)