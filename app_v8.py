from flask import Flask, request, jsonify, send_file
import subprocess
import os
import time
import re
import torch
from transformers import AutoTokenizer, M2M100ForConditionalGeneration, pipeline
from flask_cors import CORS

app = Flask(__name__)
#CORS(app, resources={r"/*": {"origins": "http://localhost:3001"}})
CORS(app, resources={r"/*": {"origins": ["http://localhost:3001", "https://yeefm.com", "https://www.yeefm.com"]}})

base_dir = "./data"  # Ensure this points to your ebook data directory
audio_files_dir = "./audio_files"  # Directory where synthesized audio files are stored

# Ensure the audio_files directory exists
os.makedirs(audio_files_dir, exist_ok=True)

# Load translation model and tokenizer once at startup
translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/nllb-200-distilled-600M", torch_dtype=torch.float16).to("cuda").eval()
translation_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

def translate_long_text(long_text, max_chunk_length, translator, max_length):
    chunks = [long_text[i:i + max_chunk_length] for i in range(0, len(long_text), max_chunk_length)]
    translated_chunks = [translator(chunk, max_length=max_length)[0] for chunk in chunks]
    translated_text = ''.join([chunk['translation_text'] for chunk in translated_chunks])
    return translated_text

def create_translator(src_lang, tgt_lang):
    return pipeline('translation', model=translation_model, tokenizer=translation_tokenizer, src_lang=src_lang, tgt_lang=tgt_lang, device="cuda")

def translate_text(text, tgt_lang):
    translator = create_translator("en_Latn", f"{tgt_lang}_Latn")
    max_chunk_length = 1000  # Adjust as needed
    max_length = 400  # Adjust as needed
    translated_text = translate_long_text(text, max_chunk_length, translator, max_length)
    return translated_text

def translate_text2(text, src_lang, tgt_lang):
    translator = create_translator(src_lang, tgt_lang)
    max_chunk_length = 1000  # Adjust as needed
    max_length = 400  # Adjust as needed
    translated_text = translate_long_text(text, max_chunk_length, translator, max_length)
    return translated_text


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    ebook_id = data.get('ebook_id')
    query = data.get('query')
    user_id = data.get('user_id')
    title = data.get('title')
    description = data.get("description", "").lower()  # Adjusted to lowercase 'description'

    if not ebook_id or not query or not user_id:
        return jsonify({"error": "Ebook ID, query, and user ID are required"}), 400

    ebook_dir = os.path.join(base_dir, str(ebook_id))

    if os.path.exists(ebook_dir):
        try:
            result = subprocess.run(['python', 'getting9.py', ebook_id, query, user_id, title, description], capture_output=True, text=True)
            return jsonify({"response": result.stdout.strip()}), 200
        except Exception as e:
            app.logger.error(f"Error in running getting9.py: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        try:
            subprocess.run(['python', 'store8.py', ebook_id], check=True)
            result = subprocess.run(['python', 'getting9.py', ebook_id, query, user_id, title, description], capture_output=True, text=True)
            return jsonify({"response": result.stdout.strip()}), 200
        except subprocess.CalledProcessError as e:
            app.logger.error(f"CalledProcessError in running store8.py or getting9.py: {e}")
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            app.logger.error(f"Error in running store8.py or getting9.py: {e}")
            return jsonify({"error": str(e)}), 500


@app.route('/lang_chat', methods=['POST'])
def lang_chat():
    data = request.json
    ebook_id = data.get('ebook_id')
    query = data.get('query')
    user_id = data.get('user_id')
    title = data.get('title')
    description = data.get("description", "").lower()  # Adjusted to lowercase 'description'
    language_code = data.get('language', 'eng')  # Default to 'eng' if not provided

    if not ebook_id or not query or not user_id:
        return jsonify({"error": "Ebook ID, query, and user ID are required"}), 400

    ebook_dir = os.path.join(base_dir, str(ebook_id))

    try:
        # Translate query to English if it's not already in English
        translated_query = query
        if language_code != 'eng':
            translated_query = translate_text2(query, f"{language_code}_Latn", "en_Latn")
            app.logger.debug(f"Translated query to English: {translated_query}")

        if os.path.exists(ebook_dir):
            result = subprocess.run(['python', 'getting9.py', ebook_id, translated_query, user_id, title, description], capture_output=True, text=True)
            response = result.stdout.strip()

            # Translate response back to the user's preferred language
            translated_response = response
            if language_code != 'eng':
                translated_response = translate_text2(response, "en_Latn", f"{language_code}_Latn")
                app.logger.debug(f"Translated response to {language_code}: {translated_response}")

            return jsonify({"response": translated_response}), 200
        else:
            subprocess.run(['python', 'store8.py', ebook_id], check=True)
            result = subprocess.run(['python', 'getting9.py', ebook_id, translated_query, user_id, title, description], capture_output=True, text=True)
            response = result.stdout.strip()

            # Translate response back to the user's preferred language
            translated_response = response
            if language_code != 'eng':
                translated_response = translate_text2(response, "en_Latn", f"{language_code}_Latn")
                app.logger.debug(f"Translated response to {language_code}: {translated_response}")

            return jsonify({"response": translated_response}), 200

    except subprocess.CalledProcessError as e:
        app.logger.error(f"CalledProcessError in subprocess: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500
           


@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    app.logger.debug(f"Received data: {data}")

    # Extract and clean data
    text = data.get('text', '')
    book_title = data.get('book_title', '')
    page_number = data.get('page_number', '')
    language = data.get('language', 'eng')  # Default to English if no language is provided

    # Clean page_number
    page_number = clean_page_number(page_number)
    print(f"page number: {page_number}")

    # Clean title
    book_title = clean_title(book_title)
    print(f"cleaned title {book_title}")
 
    if not text or not book_title or not page_number:
        return jsonify({"error": "Text, book title, and page number are required"}), 400

    try:
        # Print cleaned data
        app.logger.debug(f"Cleaned data: book_title={book_title}, page_number={page_number}, text={text}")

        # Translate text if language is not English
        if language != 'eng':
            text = translate_text(text, language)
            app.logger.debug(f"Translated text to {language}: {text}")

        # Check if the audio file already exists
        audio_file_path = os.path.join(audio_files_dir, f"{book_title}_{page_number}_{language}.wav")
        app.logger.debug(f"Checking if audio file exists: {audio_file_path}")
        
        if os.path.exists(audio_file_path):
            app.logger.debug(f"Audio file exists: {audio_file_path}")
            return jsonify({"audio_url": f"/audio/{book_title}_{page_number}_{language}.wav"}), 200
        
        # If the file doesn't exist, generate it
        cmd = ['bash', 'tts_v5.sh', book_title, page_number, text, language]
        app.logger.debug(f"Running command: {cmd}")
        subprocess.run(cmd, check=True)

        # Wait a bit to ensure the file is generated
        time.sleep(5)

        # Ensure the file was generated
        if not os.path.exists(audio_file_path):
            app.logger.error(f"Failed to generate audio file: {audio_file_path}")
            return jsonify({"error": "Failed to generate audio file"}), 500

        app.logger.debug(f"Audio file generated: {audio_file_path}")
        # Return the audio file URL
        return jsonify({"audio_url": f"/audio/{book_title}_{page_number}_{language}.wav"}), 200

    except subprocess.CalledProcessError as e:
        app.logger.error(f"CalledProcessError in running tts_v4.sh: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        app.logger.error(f"Error in synthesizing audio: {e}")
        return jsonify({"error": str(e)}), 500

def clean_page_number(page_number):
    if 'n/a' in page_number.lower():
       page_number = str(int(time.time()))
    cleaned_page_number = re.sub(r'[^a-zA-Z0-9\s]', '', page_number)
    cleaned_page_number = cleaned_page_number.replace(' ', '_')
    return cleaned_page_number.strip()

def clean_title(title):
    cleaned_title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    cleaned_title = cleaned_title.replace(' ', '_')
    return cleaned_title.strip()

@app.route('/audio/<filename>')
def serve_audio(filename):
    try:
        audio_file_path = os.path.join(audio_files_dir, filename)
        app.logger.debug(f"Serving audio file: {audio_file_path}")
        return send_file(audio_file_path, mimetype='audio/wav')
    except Exception as e:
        app.logger.error(f"Error in serving audio file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/hello')
def hello():
    return 'Hello, World!'

@app.route('/testchat')
def testchat():
    return 'test chat'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
