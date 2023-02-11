import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {"url": "https://user-images.githubusercontent.com/46135649/218277710-9a5f6e1c-051f-4f2c-852f-a21b937080a9.png"}

result = requests.post(url, json=data).json()

print(result)