from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Home page
@app.route("/")
def home():
    return render_template("home.html")


# Upload page
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            flash("File uploaded successfully!", "success")
            return redirect(url_for("analysis", filename=file.filename))
        else:
            flash("Please upload a valid file.", "danger")
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)
