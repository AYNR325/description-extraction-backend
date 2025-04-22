import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import io
import base64
import pytesseract
from gtts import gTTS
from flask import Flask, request, jsonify, send_file, Response, session
from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app)

# Required for session
# app.secret_key = "your_secret_key_here"  # Replace with a secure random key in production

# Configure Google Gemini API
GEMINI_API_KEY = "AIzaSyClwx3CyenZMAk5m9WqSw4_5ERuzXR7DCI"
genai.configure(api_key=GEMINI_API_KEY)

SUPPORTED_LANGUAGES = {
    "en": "eng",
    "hi": "hin",
    "mr": "mar"
}

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi"
}

@app.route('/set-language', methods=['POST'])
def set_language():
    """Set user preferred language using session."""
    data = request.get_json()
    language = data.get('language', 'en')
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({"error": "Unsupported language"}), 400
    session['language'] = language
    return jsonify({"message": f"Language set to {language}"}), 200

@app.route('/get-language', methods=['GET'])
def get_language():
    """Get the currently selected language."""
    lang = session.get('language', 'en')
    return jsonify({"language": lang})

def extract_text_from_image(image, lang):
    """Extract text from an image using Tesseract OCR in the specified language."""
    image_pil = Image.fromarray(image)
    tesseract_lang = SUPPORTED_LANGUAGES.get(lang, "eng")
    text = pytesseract.image_to_string(image_pil, lang=tesseract_lang)
    return text.strip() if text else "No text detected."

def describe_image(image, lang):
    """Use Google's Gemini API to generate a detailed description of an image in the specified language."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    image_pil = Image.fromarray(image)

    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    lang_name = LANGUAGE_NAMES.get(lang, "English")
    prompt_text = f"Describe this image in 8-9 sentences in {lang_name}."

    response = model.generate_content([
        {"text": prompt_text},
        {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(img_byte_arr).decode('utf-8')}}
    ], stream=False)

    return response.text if response and hasattr(response, 'text') else "Could not generate description."

def generate_speech(text, lang):
    """Convert text to speech in the specified language and return as an audio file (MP3)."""
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)
    return audio_io

@app.route('/process-image', methods=['POST'])
def process_image():
    """Processes the image and returns both text and MP3 in a single multipart response."""
    data = request.get_json()

    if 'image' not in data or 'mode' not in data:
        return jsonify({"error": "Missing 'image' or 'mode' parameter"}), 400

    try:
        # Decode Base64 image
        image_data = base64.b64decode(data['image'])
        image_np = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Determine language from request or session or fallback to 'en'
        lang = data.get('language') or session.get('language', 'en')
        if lang not in SUPPORTED_LANGUAGES:
            lang = "en"

        # Perform mode-specific processing
        if data['mode'] == "description":
            result_text = describe_image(image, lang)
        elif data['mode'] == "text":
            result_text = extract_text_from_image(image, lang)
        else:
            return jsonify({"error": "Invalid mode. Use 'description' or 'text'"}), 400

        # Generate MP3
        audio_io = generate_speech(result_text, lang)

        # Prepare multipart response
        boundary = "----MultipartBoundary"
        response_body = f"--{boundary}\r\n"
        response_body += "Content-Type: application/json\r\n\r\n"
        response_body += jsonify({"text": result_text}).data.decode("utf-8")
        response_body += f"\r\n--{boundary}\r\n"
        response_body += "Content-Type: audio/mpeg\r\n\r\n"

        def generate():
            yield response_body.encode("utf-8")
            yield audio_io.read()
            yield f"\r\n--{boundary}--\r\n".encode("utf-8")

        return Response(generate(), mimetype=f"multipart/mixed; boundary={boundary}")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Default to 8080 if PORT is not set
    app.run(host="0.0.0.0", port=port, debug=True)