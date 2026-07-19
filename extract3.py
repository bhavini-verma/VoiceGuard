import codecs
import re

path = r'C:\Users\rajas\Downloads\Redesigning knowledge base for professional standards - Claude.html'
with codecs.open(path, 'r', 'utf-8') as f:
    text = f.read()

# Search for any large script tags containing JSON
for m in re.finditer(r'<script[^>]*>(.*?)</script>', text, re.DOTALL):
    script_content = m.group(1)
    if len(script_content) > 10000:
        print("Found large script tag of size", len(script_content))
        # look for "renderKB" in the script content
        idx = script_content.find('renderKB')
        if idx != -1:
            print("Found renderKB in script tag!")
            print(script_content[max(0, idx-100):idx+500])
