
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

# Clear cached environment variables
os.environ.clear()
load_dotenv(override=True)

"""
    invoke() in AzureOpenAI assumes you are using a Text Completion API endpoint.
    If your model (gpt-4o-mini) supports chat but not text completions, this will fail with Unsupported data type.
    Switch to AzureChatOpenAI for chat-compatible models (e.g., GPT-4 or GPT-3.5-turbo). It is specifically designed for Azure OpenAI models that use the Chat Completion API.
    
    from langchain_openai import AzureOpenAI
    model = AzureOpenAI(
        deployment_name="gpt-4o-mini"
    )
"""

# Initialize Azure OpenAI model with LangChain
model = AzureChatOpenAI(
    deployment_name="gpt-4o-mini"
)

messages = [
    SystemMessage("Translate the following from English into Korean"),
    HumanMessage("Hi!"),
]
response = model.invoke(messages)
print(response.content)




# # Initialize Azure OpenAI client with openai package
# from openai import AzureOpenAI

# client = AzureOpenAI()
# print("======= client =======>>")
# print(client)
# print("======= client =======<<")
# response = client.chat.completions.create(
#     model="gpt-4o-mini",  # Replace with your deployment name
#     messages=[
#         {"role": "system", "content": "Translate the following from English into Korean"},
#         {"role": "user", "content": "Hi! My name is Jehyun"}
#     ]
# )
# print(response.choices[0].message.content)