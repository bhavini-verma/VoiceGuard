import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Inject window.onerror at the very top of the head
injection = """
<script>
  window.onerror = function(msg, url, line, col, error) {
      const errDiv = document.createElement('div');
      errDiv.style.position = 'fixed';
      errDiv.style.top = '0';
      errDiv.style.left = '0';
      errDiv.style.zIndex = '999999';
      errDiv.style.background = 'red';
      errDiv.style.color = 'white';
      errDiv.style.padding = '10px';
      errDiv.style.fontSize = '16px';
      errDiv.innerHTML = "<b>JS ERROR:</b> " + msg + " (Line " + line + ")";
      document.body.appendChild(errDiv);
      return false;
  };
</script>
"""

idx = text.find("<head>")
if idx != -1:
    text = text[:idx+6] + injection + text[idx+6:]
    with codecs.open(path, "w", "utf-8") as f:
        f.write(text)
    print("Injected error catcher.")
else:
    print("Could not find <head>")
