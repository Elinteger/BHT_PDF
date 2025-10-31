import cv2
import numpy as np
import os
import pandas as pd
import pymupdf
import tempfile
from autogluon.multimodal import MultiModalPredictor
from deskew import determine_skew
from dotenv import load_dotenv
from pathlib import Path
from skimage.color import rgb2gray
from skimage.transform import rotate

# FIXME: Entire code works, but without using PIL, may be a bit hacky at times because of that
# using PIL was avoided as it is an external dependency that might pose as a problem in deployment

def init_tools():
    global pred, A4_W, A4_H
    load_dotenv()
    CLASSIFICATION_MODEL = os.getenv('CLASSIFICATION_MODEL')
    pred = MultiModalPredictor.load(CLASSIFICATION_MODEL)
    A4_W, A4_H = 595, 842


def pdf_to_imgs(pdf_bytes):
    #TODO: check if code can be compressed (unnecessary saves etc)
    doc = pymupdf.open('pdf', pdf_bytes)
    scaled_doc = pymupdf.open()
    # TODO: re-think on how to handle in-memory images, maybe use PIL (avoided due to it being an external dependency) -> see FIXME:
    pixmaps = []
    image_arr = []  # List 3d-numpy array with RGB images in range 0-255 (width (x), height (y), channels=3 -> RGB)

    # normalize scale of PDF-Pages to A4 and remove rotation
    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        w, h = page.rect.width, page.rect.height
        rotation = page.rotation
        if rotation != 0:
            page.set_rotation(0)
        a4_scale = min(A4_W / w, A4_H / h)
        new_w, new_h = w * a4_scale, h * a4_scale
        dx, dy = (A4_W - new_w) / 2, (A4_H - new_h) / 2
        new_page = scaled_doc.new_page(width=A4_W, height=A4_H)
        # place original scaled page onto new empty page
        trans = pymupdf.Matrix(a4_scale, a4_scale).pretranslate(dx / a4_scale, dy / a4_scale)
        new_page.show_pdf_page(new_page.rect, doc, page_idx, trans)

    # turn scaled pdf into images
    for page in scaled_doc:
        pixmaps.append(page.get_pixmap(dpi=200, colorspace='rgb', alpha=False))
    
    # descew images 
    for pix in pixmaps:
        # turn pixmap into 3d numpy array
        pix_np = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, -1))
        grayscale = rgb2gray(pix_np)
        skew = determine_skew(grayscale)
        if skew != 0.0:  # skip if no skew to save time
            image_arr.append(pix_np)
            image_arr.append(rotate(pix_np, skew, resize=False, preserve_range=True).astype(np.uint8))
        else:
            image_arr.append(pix_np)
            
    return scaled_doc, image_arr
   

def extract_relevant_pages(scaled_doc, img_arr) -> bytes:
    # Temporary directory for JPEGs <- TODO: Code written by ChatGPT, check!
    with tempfile.TemporaryDirectory() as tmpdir:
        img_paths = []
        for i, img in enumerate(img_arr):
            path = Path(tmpdir) / f"{i}.jpg"
            # OpenCV expects images in uint8 and in BGR format
            img_uint8 = img.astype(np.uint8)
            if img_uint8.shape[2] == 3:  # Convert RGB to BGR for OpenCV
                img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), img_uint8)
            img_paths.append(str(path))

        y_pred = pred.predict(img_paths)

    idx_dict = {}
    idx_dict["Stammdaten"] = np.where(y_pred == 'Stammdaten')[0].tolist()
    idx_dict["Kursübersichten" ] = np.where(y_pred == 'Kursübersicht')[0].tolist()
    idx_dict["Zeugnisse"] = np.where(y_pred == 'Zeugnis')[0].tolist()

    # create pdf
    ## misc stuff needed for doc creation
    p = pymupdf.Point(50, 842/2)
    line_start = pymupdf.Point(p.x-10, p.y-25)   
    line_end   = pymupdf.Point(p.x-10, p.y+5) 
    toc = []  # level, title, page

    doc = pymupdf.open()
    
    for name, idxs in idx_dict.items():
        # create cover page
        textpage = doc.new_page() 
        textpage.draw_line(line_start, line_end, width=3.5)
        textpage.insert_text(
            point=p,
            text=name,
            fontsize=30,
        )
        toc.append([1, name, doc.page_count])
        for idx in idxs:
            doc.insert_pdf(docsrc=scaled_doc, from_page=idx, to_page=idx)

    doc.set_toc(toc=toc)
    
    return doc.convert_to_pdf()
