# Import library yang diperlukan
from flask import Flask, render_template, request, redirect, url_for, session
from PIL import Image, ImageEnhance, ImageFilter  # Pillow untuk manipulasi gambar
import cv2  # OpenCV untuk pengolahan gambar tingkat lanjut
import os  # Untuk operasi file dan direktori
import numpy as np  # Untuk manipulasi array, digunakan bersama OpenCV
from edge_detection import save_edges  # Import fungsi deteksi tepi
import matplotlib.pyplot as plt  # Untuk membuat histogram

# Inisialisasi aplikasi Flask
app = Flask(__name__)
app.secret_key = 'bebas_isi_apa_saja'  # Secret key untuk session

# Konfigurasi folder untuk menyimpan file yang diunggah
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Fungsi untuk menyimpan hasil morfologi dan histogram
def save_morphology_and_histogram(filename, upload_folder):
    path = os.path.join(upload_folder, filename)
    img = cv2.imread(path, 0)
    kernel = np.ones((5,5), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    img_dilation = cv2.dilate(img, kernel, iterations=1)

    # Simpan hasil erosi dan dilasi
    erosion_name = f"{os.path.splitext(filename)[0]}_erosion.jpg"
    dilation_name = f"{os.path.splitext(filename)[0]}_dilation.jpg"
    cv2.imwrite(os.path.join(upload_folder, erosion_name), img_erosion)
    cv2.imwrite(os.path.join(upload_folder, dilation_name), img_dilation)

    # Simpan histogram input
    plt.figure(figsize=(4,3))
    plt.hist(img.ravel(), bins=256)
    plt.title('Histogram Citra Input')
    plt.tight_layout()
    hist_input_name = f"{os.path.splitext(filename)[0]}_hist_input.jpg"
    plt.savefig(os.path.join(upload_folder, hist_input_name))
    plt.close()

    # Simpan histogram erosi
    plt.figure(figsize=(4,3))
    plt.hist(img_erosion.ravel(), bins=256)
    plt.title('Histogram Citra Erosi')
    plt.tight_layout()
    hist_erosion_name = f"{os.path.splitext(filename)[0]}_hist_erosion.jpg"
    plt.savefig(os.path.join(upload_folder, hist_erosion_name))
    plt.close()

    # Simpan histogram dilasi
    plt.figure(figsize=(4,3))
    plt.hist(img_dilation.ravel(), bins=256)
    plt.title('Histogram Citra Dilasi')
    plt.tight_layout()
    hist_dilation_name = f"{os.path.splitext(filename)[0]}_hist_dilation.jpg"
    plt.savefig(os.path.join(upload_folder, hist_dilation_name))
    plt.close()

    return erosion_name, dilation_name, hist_input_name, hist_erosion_name, hist_dilation_name

# Route untuk halaman utama
@app.route('/')
def index():
    results = session.pop('results', None)
    return render_template('index.html', results=results)

# Route untuk menangani unggahan file
@app.route('/upload', methods=['POST'])
def upload():
    # Memeriksa apakah file ada dalam request
    if 'files[]' not in request.files:
        # Jika tidak ada file, redirect ke halaman utama
        return redirect(url_for('index'))

    # Mengambil semua file dari request
    files = request.files.getlist('files[]')
    results = []

    # Memproses setiap file
    for file in files:
        if file and file.filename != '':
            # Menyimpan file asli
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Baca gambar dengan OpenCV
            img = cv2.imread(filepath)
            
            # Deteksi warna biru
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            blue1 = np.array([110, 50, 50])
            blue2 = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, blue1, blue2)
            blue_detection = cv2.bitwise_and(img, img, mask=mask)
            blue_filename = f"blue_{file.filename}"
            blue_filepath = os.path.join(app.config['UPLOAD_FOLDER'], blue_filename)
            cv2.imwrite(blue_filepath, blue_detection)

            # Tambahkan proses opening pada mask biru
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            opening_filename = f"opening_{file.filename}"
            opening_filepath = os.path.join(app.config['UPLOAD_FOLDER'], opening_filename)
            cv2.imwrite(opening_filepath, opening)

            # Memproses gambar untuk meningkatkan kualitasnya
            img_pil = Image.open(filepath)
            img_pil = improve_image_quality(img_pil)

            # Menentukan nama file hasil yang telah diproses
            processed_filename = 'processed_' + file.filename
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)

            # Menyimpan gambar yang telah diproses
            img_pil.save(processed_filepath)

            # Melakukan deteksi tepi
            save_edges(file.filename, app.config['UPLOAD_FOLDER'])

            # Simpan hasil morfologi dan histogram
            erosion_name, dilation_name, hist_input_name, hist_erosion_name, hist_dilation_name = save_morphology_and_histogram(file.filename, app.config['UPLOAD_FOLDER'])

            # Mengumpulkan hasil pemrosesan untuk file ini
            result = {
                'original': file.filename,
                'processed': processed_filename,
                'blue_detection': blue_filename,
                'opening': opening_filename,
                'canny': f"{os.path.splitext(file.filename)[0]}_canny.jpg",
                'sobel': f"{os.path.splitext(file.filename)[0]}_sobel.jpg",
                'laplacian': f"{os.path.splitext(file.filename)[0]}_laplacian.jpg",
                'erosion': erosion_name,
                'dilation': dilation_name,
                'hist_input': hist_input_name,
                'hist_erosion': hist_erosion_name,
                'hist_dilation': hist_dilation_name
            }
            results.append(result)

    session['results'] = results
    return redirect(url_for('index'))

# Fungsi untuk meningkatkan kualitas gambar
def improve_image_quality(img):
    # Mengonversi gambar menjadi grayscale
    img = img.convert('L')

    # Meningkatkan kontras gambar
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)

    # Mengonversi gambar ke format OpenCV untuk pengolahan lebih lanjut
    img = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)
    # Menghaluskan gambar menggunakan Gaussian Blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Mengonversi kembali gambar ke format Pillow
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    # Menajamkan gambar menggunakan filter Pillow
    img = img.filter(ImageFilter.SHARPEN)

    # Mengembalikan gambar yang telah diproses
    return img

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    # Menjalankan server Flask dalam mode debug
    app.run(debug=True)