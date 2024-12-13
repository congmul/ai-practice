import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

from langchain_openai import ChatOpenAI


# If set API key inside OPENAI_API_KEY variable, then ChatOpenAI func get the value automatically.
# So you don't need to pass the api key directly to ChatOpenAI
# openai_api_key = os.environ.get('OPENAI_API_KEY')
model = ChatOpenAI(model="gpt-4o-mini")

# messages = [
#     SystemMessage("Translate the following from English into Italian"),
#     HumanMessage("hi!"),
# ]
# model.invoke(messages)
# for token in model.stream(messages):
#     print(token.content, end="|")

system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "Korean", "text": "hi!"})

print(prompt)
print(prompt.to_messages())

response = model.invoke(prompt)
print(response.content)