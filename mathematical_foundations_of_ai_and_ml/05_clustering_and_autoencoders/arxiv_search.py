import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

query = 'all:"autoencoder" AND all:"materials" AND all:"transfer learning"'
url = f'http://export.arxiv.org/api/query?search_query={urllib.parse.quote(query)}&start=0&max_results=5'
try:
    response = urllib.request.urlopen(url, timeout=10)
    xml_data = response.read()
    root = ET.fromstring(xml_data)

    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        print("Title:", entry.find("{http://www.w3.org/2005/Atom}title").text)
        print("ID:", entry.find("{http://www.w3.org/2005/Atom}id").text)
        print("Summary:", entry.find("{http://www.w3.org/2005/Atom}summary").text[:200].replace('\n', ' '), "...\n")
except Exception as e:
    print(f"Error: {e}")
