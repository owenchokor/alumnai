from flask import Flask, render_template, request, send_file, jsonify
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader, PdfWriter
from pydub import AudioSegment
import shutil
import sys
folder_path = "C:\\Users\\82103\\Desktop\\pdf_task\\coding_team"
if folder_path not in sys.path:
    sys.path.append(folder_path)
os.environ["PATH"] += os.pathsep + folder_path
print(os.environ["PATH"])
import src.annot as annot

app = Flask(__name__)

# Directory to temporarily store files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Directory to store downloadable files
DOWNLOAD_FOLDER = 'downloads'
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

# Directory to store audio files
AUDIO_FOLDER = 'audios'
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER

# Create necessary folders if they don't exist
for folder in [UPLOAD_FOLDER, DOWNLOAD_FOLDER, AUDIO_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def process_pdf(pdf_path, mp3_path, output_pdf_path):
    """For testing: just return the exact input PDF."""
    try:
        """# Copy the input PDF to the output location without modification
        annot.annot.add_summary_to_pdf('../../ex_33.pdf', '../../alumnai/results/241025.json', output_pdf_path)"""
        shutil.copy(pdf_path, output_pdf_path)
        return True
    except Exception as e:
        print(f"Error copying PDF: {e}")
        return False

# Route to render the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle PDF and audio upload and processing
@app.route('/upload/process', methods=['POST'])
def upload_and_process():
    if 'pdf_file' not in request.files or 'audio_file' not in request.files:
        return jsonify({"message": "PDF or audio file missing"}), 400
    
    pdf_file = request.files['pdf_file']
    audio_file = request.files['audio_file']
    
    # Validate the PDF and audio file types
    if pdf_file.filename == '' or not pdf_file.filename.endswith('.pdf'):
        return jsonify({"message": "Invalid PDF file"}), 400
    if audio_file.filename == '' or not audio_file.filename.endswith(('.mp3', '.wav', '.m4a')):
        return jsonify({"message": "Invalid audio file"}), 400
    
    # Secure and save the uploaded files
    pdf_filename = secure_filename(pdf_file.filename)
    audio_filename = secure_filename(audio_file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    audio_path = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
    
    pdf_file.save(pdf_path)
    audio_file.save(audio_path)
    
    # Output path for the processed PDF
    output_pdf_path = os.path.join(app.config['DOWNLOAD_FOLDER'], f"processed_{pdf_filename}")
    
    # Process the PDF and audio together
    if process_pdf(pdf_path, audio_path, output_pdf_path):
        download_link = f"/download/{os.path.basename(output_pdf_path)}"
        return jsonify({"message": "PDF processed successfully", "download_link": download_link}), 200
    else:
        return jsonify({"message": "Error processing PDF"}), 500

# Route to download files
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True)
