import requests

url  = "https://0ae0s4rn1g.execute-api.us-west-2.amazonaws.com/test"

data = {"url": "https://user-images.githubusercontent.com/46135649/218277710-9a5f6e1c-051f-4f2c-852f-a21b937080a9.png"}

result = requests.post(url, json=data).json()

print(result)