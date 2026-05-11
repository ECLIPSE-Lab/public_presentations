import urllib.request
import json
import urllib.parse

query = "autoencoder transfer learning materials property prediction"
url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(query)}&limit=3&fields=title,authors,year,abstract,url,openAccessPdf"
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as response:
        data = json.loads(response.read().decode())
        for paper in data.get('data', []):
            print("Title:", paper.get('title'))
            print("Year:", paper.get('year'))
            print("URL:", paper.get('url'))
            print("PDF:", paper.get('openAccessPdf'))
            print("Abstract:", str(paper.get('abstract'))[:200])
            print("---")
except Exception as e:
    print(f"Error: {e}")
