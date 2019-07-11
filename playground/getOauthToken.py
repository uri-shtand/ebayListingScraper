from oauth2client.client import OAuth2WebServerFlow
import json

with open('application_token.txt') as json_file:
    appTokenData = json.load(json_file)
    CLIENT_ID = appTokenData['clientId']
    CLIENT_SECRET = appTokenData['clientSecret']
    SCOPES = ("https://api.ebay.com/oauth/api_scope",)
    flow = OAuth2WebServerFlow(CLIENT_ID, CLIENT_SECRET, " ".join(SCOPES))
