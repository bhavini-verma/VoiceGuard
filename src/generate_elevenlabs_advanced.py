import os
import requests
import json
import time
import random
import wave

API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
if not API_KEY:
    print("Warning: ELEVENLABS_API_KEY environment variable not set.")
HEADERS = {
    "xi-api-key": API_KEY,
    "Content-Type": "application/json"
}

# 70 Banking templates
banking_prefixes = [
    "नमस्ते सर, मैं एचडीएफसी बैंक से बात कर रहा हूँ।",
    "गुड मॉर्निंग, मैं एसबीआई क्रेडिट कार्ड डिपार्टमेंट से बोल रहा हूँ।",
    "हेलो, क्या आपको मेरी आवाज़ आ रही है? मैं आईसीआईसीआई बैंक से हूँ।",
    "सर, आपके बैंक खाते के संबंध में एक अत्यंत आवश्यक कॉल है।",
    "नमस्ते, मैं आपके बैंक का रिलेशनशिप मैनेजर बोल रहा हूँ।",
    "कृपया ध्यान दें, यह कॉल आपके बैंक खाते की सुरक्षा के लिए है।",
    "हेलो सर, क्या मैं आपसे दो मिनट बात कर सकता हूँ? यह आपके बैंक खाते से जुड़ा है।"
]

banking_bodies = [
    "आपके खाते से अभी पच्चीस हज़ार रुपये का संदिग्ध लेन-देन हुआ है।",
    "आपके क्रेडिट कार्ड की लिमिट बढ़ाने का एक नया ऑफर आया है, जो केवल आज के लिए है।",
    "आपके खाते का केवाईसी (KYC) अपडेट पेंडिंग है, जिसकी वजह से आपका खाता ब्लॉक हो सकता है।",
    "हमने देखा है कि आपके खाते से विदेश में कोई ट्रांजेक्शन करने की कोशिश की जा रही है।",
    "आपके लिए एक पूर्व-स्वीकृत पर्सनल लोन का ऑफर है, जो बहुत कम ब्याज दर पर उपलब्ध है।",
    "आपके एटीएम कार्ड की वैधता समाप्त हो रही है, इसे रिन्यू करने के लिए हमें कुछ जानकारी चाहिए।",
    "आपके खाते में कुछ असामान्य गतिविधियां पाई गई हैं, हमें इसे सुरक्षित करने के लिए आपकी सहायता चाहिए।"
]

banking_suffixes = [
    "कृपया इस लेन-देन को रोकने के लिए तुरंत अपना वेरिफिकेशन पूरा करें।",
    "इस ऑफर का लाभ उठाने के लिए, कृपया अपना रजिस्टर्ड मोबाइल नंबर कंफर्म करें।",
    "कृपया अपना वन-टाइम पासवर्ड (OTP) शेयर करें ताकि हम प्रक्रिया को पूरा कर सकें।",
    "क्या आप मुझे अपनी जन्मतिथि और पैन कार्ड का विवरण बता सकते हैं?",
    "कृपया लाइन पर बने रहें, मैं आपको हमारे सुरक्षा अधिकारी से कनेक्ट कर रहा हूँ।",
    "अगर आपने यह ट्रांजेक्शन नहीं किया है, तो तुरंत इस कॉल के माध्यम से रिपोर्ट करें।",
    "कृपया मुझे अपना आधार नंबर कंफर्म करें ताकि आपका खाता सुचारू रूप से चलता रहे।"
]

# 30 General templates
general_prefixes = [
    "नमस्ते, मैं रिलायंस जियो से बात कर रहा हूँ।",
    "हेलो सर, मैं एक ऑनलाइन सर्वे कंपनी से बोल रहा हूँ।",
    "गुड मॉर्निंग, मैं मेक-माय-ट्रिप से बात कर रहा हूँ।",
    "नमस्ते, क्या मैं राहुल जी से बात कर रहा हूँ?"
]

general_bodies = [
    "आपके नंबर पर एक विशेष रिचार्ज ऑफर उपलब्ध है जिसमें आपको रोज़ाना २ जीबी डेटा मिलेगा।",
    "क्या आप हाल ही में किसी यात्रा पर गए थे? हम आपका फीडबैक जानना चाहते हैं।",
    "आपके लिए हमारे नए हॉलिडे पैकेज पर भारी छूट उपलब्ध है।",
    "हम ग्राहकों के अनुभव को बेहतर बनाने के लिए एक छोटा सा सर्वे कर रहे हैं।"
]

general_suffixes = [
    "क्या आप मुझे अपने अनुभव के बारे में कुछ और बता सकते हैं?",
    "ऑफर को एक्टिवेट करने के लिए कृपया एक दबाएं।",
    "क्या मैं आपको इस पैकेज की विस्तृत जानकारी आपके व्हाट्सएप पर भेज दूँ?",
    "अपना कीमती समय देने के लिए बहुत-बहुत धन्यवाद।"
]

def generate_hindi_sentences():
    random.seed(42)
    sentences = []
    
    for _ in range(70):
        p = random.choice(banking_prefixes)
        b = random.choice(banking_bodies)
        s = random.choice(banking_suffixes)
        sentences.append(f"{p} {b} {s}")
        
    for _ in range(30):
        p = random.choice(general_prefixes)
        b = random.choice(general_bodies)
        s = random.choice(general_suffixes)
        sentences.append(f"{p} {b} {s}")
        
    random.shuffle(sentences)
    return sentences

def get_voices():
    response = requests.get("https://api.elevenlabs.io/v1/voices", headers=HEADERS)
    if response.status_code == 200:
        voices = response.json().get('voices', [])
        # Extract at least 10 different voices to ensure diversity (Elderly, Hoarse, Emotional will be naturally varied)
        return [v['voice_id'] for v in voices[:10]]
    else:
        print("Error fetching voices:", response.text)
        return []

def generate_audio(text, voice_id, output_path):
    import librosa
    import soundfile as sf
    import tempfile

    # Request MP3 (works on all tiers)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128"
    
    # Varying styles: randomly tweak stability and similarity to get different emotions (urgent vs formal)
    stability = round(random.uniform(0.3, 0.8), 2)
    similarity = round(random.uniform(0.5, 0.9), 2)
    
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity,
            "style": random.uniform(0.0, 0.5) # Adds a bit of dynamic speaking style
        }
    }
    
    response = requests.post(url, json=payload, headers=HEADERS)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
            temp_mp3.write(response.content)
            temp_mp3_path = temp_mp3.name
        
        try:
            # Convert MP3 to uncompressed 16kHz WAV natively
            y, sr = librosa.load(temp_mp3_path, sr=16000, mono=True)
            sf.write(output_path, y, sr, subtype='PCM_16')
            return True
        except Exception as e:
            print(f"Error converting to WAV for {voice_id}: {e}")
            return False
        finally:
            os.remove(temp_mp3_path)
    else:
        print(f"Error generating audio for {voice_id}: {response.text}")
        return False

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'data', 'raw_fake', 'fake')
    os.makedirs(output_dir, exist_ok=True)
    
    sentences = generate_hindi_sentences()
    print("Fetching available voices...")
    voices = get_voices()
    if not voices:
        print("No voices found, falling back to default voice ID.")
        voices = ["21m00Tcm4TlvDq8ikWAM", "TxGEqnHWrfWFTfGW9XjX", "VR6AewLTigWG4xSOukaG"]
        
    print(f"Generating uncompressed WAVs using {len(voices)} voices...")
    
    count = 0
    target = 100
    
    for i in range(target):
        text = sentences[i]
        voice_id = voices[i % len(voices)]
        
        # New nomenclature for clean WAVs
        output_path = os.path.join(output_dir, f"elevenlabs_advanced_clean_{i:03d}.wav")
        
        if os.path.exists(output_path):
            print(f"Skipping {output_path}, already exists.")
            count += 1
            continue
            
        print(f"Generating {i+1}/{target} (Voice: {voice_id})...")
        success = generate_audio(text, voice_id, output_path)
        
        if success:
            count += 1
            
        time.sleep(1) # Simple rate limiting
        
    print(f"Successfully generated {count} Clean ElevenLabs Hindi clips!")

if __name__ == "__main__":
    main()
