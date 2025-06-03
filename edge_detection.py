import cv2
import os

def apply_edge_detection(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny
    edges_canny = cv2.Canny(blur, 50, 150)

    # Sobel
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    edges_sobel = cv2.magnitude(sobelx, sobely)
    edges_sobel = cv2.convertScaleAbs(edges_sobel)

    # Laplacian
    edges_laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    edges_laplacian = cv2.convertScaleAbs(edges_laplacian)

    return edges_canny, edges_sobel, edges_laplacian

def save_edges(filename, output_folder='static/uploads'):
    edges_canny, edges_sobel, edges_laplacian = apply_edge_detection(os.path.join(output_folder, filename))
    name, _ = os.path.splitext(filename)

    cv2.imwrite(os.path.join(output_folder, f'{name}_canny.jpg'), edges_canny)
    cv2.imwrite(os.path.join(output_folder, f'{name}_sobel.jpg'), edges_sobel)
    cv2.imwrite(os.path.join(output_folder, f'{name}_laplacian.jpg'), edges_laplacian) 