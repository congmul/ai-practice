from dotenv import load_dotenv
import os

from azure.keyvault.secrets import SecretClient
from azure.identity import ClientSecretCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

load_dotenv()
ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
key_vault_name = os.getenv('KEY_VAULT')
app_tenant = os.getenv('TENANT_ID')
app_id = os.getenv('APP_ID')
app_password = os.getenv('APP_PASSWORD')

# Get Azure AI services key from keyvault using the service principal credentials
key_vault_uri = f"https://{key_vault_name}.vault.azure.net/"
credential = ClientSecretCredential(app_tenant, app_id, app_password)
keyvault_client = SecretClient(key_vault_uri, credential)
secret_key = keyvault_client.get_secret("AI-Services-Key")
cog_key = secret_key.value

client = ImageAnalysisClient(
    endpoint=ai_endpoint,
    credential=AzureKeyCredential(cog_key)
)
# client.analyze(
    # image_url="https://recreationalsports.blob.core.windows.net/profileteamlogo/1732646682261-test-img.png",
    # visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ],
    # gender_neutral_caption=True,
    # language="en"
# )
result = client.analyze_from_url(
    image_url="https://recreationalsports.blob.core.windows.net/profileteamlogo/AdobeStock_472119374.webp",
    visual_features=[VisualFeatures.READ,],
    # visual_features=[VisualFeatures.READ, VisualFeatures.TAGS, VisualFeatures.PEOPLE, VisualFeatures.SMART_CROPS, VisualFeatures.CAPTION],
    gender_neutral_caption=True,
    language="en",
)
print(result)