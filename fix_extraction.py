import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGaurd_TelephonyExp\src\extract_ota_features.py"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

old_block = """                batch_audio.append(y)
                batch_meta.append({'Filename': os.path.basename(file_path), 'Label': args.label})
                
                # Extract 101 bio features using original repo logic
                bio_feats = extract_bio.extract_bio_features_chunk(y, 16000)
                batch_bio.append(bio_feats)"""

new_block = """                # Extract 101 bio features using original repo logic
                bio_feats = extract_bio.extract_bio_features_chunk(y, 16000)
                
                # Append only if bio extraction succeeds
                batch_bio.append(bio_feats)
                batch_audio.append(y)
                batch_meta.append({'Filename': os.path.basename(file_path), 'Label': args.label})"""

text = text.replace(old_block, new_block)

with codecs.open(path, "w", "utf-8") as f:
    f.write(text)

print("Fixed extraction script indexing bug")
