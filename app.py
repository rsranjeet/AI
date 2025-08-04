import uuid
import os
from flask import Flask, render_template, request, send_from_directory, jsonify
from pathlib import Path
import whisper

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Whisper model
model = whisper.load_model("base")  # or "tiny" for faster load

# Helper to format timestamps for .srt
def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

# Core transcription logic
def transcribe_to_srt_and_txt(file_path: Path):
    result = model.transcribe(str(file_path), fp16=False)

    txt_path = file_path.with_suffix(".txt")
    srt_path = file_path.with_suffix(".srt")

    # Save plain text
    with open(txt_path, "w", encoding="utf-8") as f_txt:
        f_txt.write(result["text"])

    # Save subtitles in .srt format
    with open(srt_path, "w", encoding="utf-8") as f_srt:
        for i, segment in enumerate(result["segments"], start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f_srt.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    return txt_path.name, srt_path.name

# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            save_path = Path(UPLOAD_FOLDER) / filename
            file.save(save_path)

            txt_file, srt_file = transcribe_to_srt_and_txt(save_path)

            return jsonify({
                "txt": txt_file,
                "srt": srt_file
            })

    return render_template("index.html")

# File download route
@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

# Run the server
if __name__ == "__main__":
    # app.run(debug=True)
    app.run(debug=False, use_reloader=False)

