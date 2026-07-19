import codecs
import re
import html

path = r'C:\Users\rajas\Downloads\Redesigning knowledge base for professional standards - Claude.html'
with codecs.open(path, 'r', 'utf-8') as f:
    text = f.read()

# find the block containing 'page-kb'
start_idx = text.find('id="page-kb"')
if start_idx != -1:
    print('Found page-kb at:', start_idx)
    # print 500 chars around it
    print(text[start_idx-100:start_idx+900])
else:
    print('Did not find page-kb')

js_idx = text.find('const KB_DATA =')
if js_idx != -1:
    print('Found KB_DATA at:', js_idx)
    print(text[js_idx:js_idx+500])

css_idx = text.find('.kb-category-pills')
if css_idx != -1:
    print('Found CSS at:', css_idx)
    print(text[css_idx:css_idx+500])
