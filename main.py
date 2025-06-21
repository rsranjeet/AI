import whisper
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from pathlib import Path

# Load Whisper model once
model = whisper.load_model("base")

# Function to transcribe selected file
def transcribe_file():
    file_path = filedialog.askopenfilename(
        title="Select Audio/Video File",
        filetypes=[("Media Files", "*.mp3 *.wav *.mp4 *.mkv *.mov *.flv *.aac *.m4a")]
    )
    if not file_path:
        return

    out_path = Path(file_path).with_suffix(".txt")
    status_label.config(text=f"Transcribing: {os.path.basename(file_path)}")

    try:
        result = model.transcribe(file_path, fp16=False)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        messagebox.showinfo("Success", f"Transcription saved:\n{out_path}")
        status_label.config(text="Done!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to transcribe:\n{str(e)}")
        status_label.config(text="Error occurred.")

# Set up the GUI
root = tk.Tk()
root.title("Whisper Transcriber")
root.geometry("400x200")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(expand=True)

label = tk.Label(frame, text="Select an audio/video file to transcribe", font=("Arial", 12))
label.pack(pady=10)

transcribe_button = tk.Button(frame, text="Select File & Transcribe", command=transcribe_file, font=("Arial", 12), bg="#4CAF50", fg="white")
transcribe_button.pack(pady=10)

status_label = tk.Label(frame, text="", font=("Arial", 10))
status_label.pack()

root.mainloop()
