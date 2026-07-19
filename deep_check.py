import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# 1. Check if audioBufferToWav exists
idx_abw = text.find("function audioBufferToWav")
print("1. audioBufferToWav defined:", idx_abw != -1, "at", idx_abw)

# 2. Check micChunks initialization
idx_mc = text.find("micChunks")
while idx_mc != -1 and idx_mc < 300000:
    line_s = max(0, text.rfind("\n", 0, idx_mc))
    line_e = text.find("\n", idx_mc)
    print("   micChunks at", idx_mc, ":", text[line_s:line_e].strip())
    idx_mc = text.find("micChunks", idx_mc+1)

# 3. Count decodeAudioData calls 
count = text.count("decodeAudioData")
print("2. decodeAudioData calls:", count)

# 4. Check resizeSpec definition
idx_rs = text.find("function resizeSpec")
print("3. resizeSpec at:", idx_rs)
if idx_rs != -1:
    print("   ", text[idx_rs:text.find("}", idx_rs)+1])

# 5. Check wv-stat element
idx_wvs = text.find("id=\"wv-stat\"")
print("4. wv-stat element at:", idx_wvs)
