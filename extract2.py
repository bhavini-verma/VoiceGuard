import codecs
import re

path = r'C:\Users\rajas\Downloads\Redesigning knowledge base for professional standards - Claude.html'
with codecs.open(path, 'r', 'utf-8') as f:
    text = f.read()

# find any large javascript strings or react props containing the code
# Let's search for "renderKB" ignoring case and see the context
matches = [m.start() for m in re.finditer(re.escape('renderKB'), text, re.IGNORECASE)]
print(f"Found {len(matches)} occurrences of renderKB")
for idx in matches[:5]:
    start = max(0, idx - 100)
    end = min(len(text), idx + 200)
    print("--- Context around", idx)
    print(text[start:end])

# Search for "horizontal pill"
pill_matches = [m.start() for m in re.finditer('horizontal pill', text, re.IGNORECASE)]
print(f"Found {len(pill_matches)} occurrences of 'horizontal pill'")
for idx in pill_matches[:5]:
    start = max(0, idx - 100)
    end = min(len(text), idx + 200)
    print("--- Context around", idx)
    print(text[start:end])
