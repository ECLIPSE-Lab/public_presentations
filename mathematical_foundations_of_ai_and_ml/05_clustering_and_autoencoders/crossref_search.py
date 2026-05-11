import urllib.request
import json
import urllib.parse

query = urllib.parse.quote("autoencoder materials property prediction downstream")
url = f"https://api.crossref.org/works?query={query}&select=title,author,DOI,URL,abstract&rows=3"
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'mailto:test@example.com'})
    with urllib.request.urlopen(req, timeout=10) as response:
        data = json.loads(response.read().decode())
        for item in data.get('message', {}).get('items', []):
            print("Title:", item.get('title', [''])[0])
            print("DOI:", item.get('DOI'))
            print("URL:", item.get('URL'))
            print("Abstract:", item.get('abstract', '')[:200])
            print("---")
except Exception as e:
    print(f"Error: {e}")
