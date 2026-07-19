import codecs
import re
import html

path = r'C:\Users\rajas\Downloads\Redesigning knowledge base for professional standards - Claude.html'
with codecs.open(path, 'r', 'utf-8') as f:
    text = f.read()

# Look for standard pre>code blocks
code_snippets = re.findall(r'<pre[^>]*>.*?<code[^>]*>(.*?)</code>.*?</pre>', text, re.DOTALL)
print('Found', len(code_snippets), 'pre->code blocks')
for i, c in enumerate(code_snippets):
    print('Block', i, len(c), 'chars')
    if len(c) > 0:
        c_un = html.unescape(c)
        print(c_un[:500])
        print('----------------')

# Also Claude artifacts might use "data-content" or some JSON string. 
# Let's see if we can find any HTML string starting with <div class="kb-category-pills"
kb_idx = text.find('kb-category-pills')
if kb_idx != -1:
    print('FOUND kb-category-pills at', kb_idx)
else:
    print('Did not find kb-category-pills')

# What about the wireframe layout mentioned in the prompt?
layout_idx = text.find('Breadcrumb: Support / Knowledge Base')
if layout_idx != -1:
    print('Found wireframe at:', layout_idx)

# Find any HTML code that Claude wrote
# Claude might have written an HTML artifact
artifact_idx = text.find('<!DOCTYPE html>')
# well the page itself is <!DOCTYPE html>
for m in re.finditer(r'&lt;div id=&quot;page-kb&quot;', text):
    print('Found escaped page-kb at:', m.start())

