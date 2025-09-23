from datetime import datetime, timedelta
import json
import re
from PIL import Image as PILImage
from collections import Counter
import tempfile
import subprocess
import os
from pdf2image import convert_from_path
from IPython.display import Image, display
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from typing_extensions import final
import shutil
import time
from typing import Optional, Any, Dict

def generate_unique_filename(prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{timestamp}.png"

# Example usage
filename = generate_unique_filename("table")


def crop_image(png_path, show_img=False):
    # ---------- 1. Pre-process ----------
    img  = cv2.imread(png_path)
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = lab[:, :, 0]                                # luminance only
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 31, 15)

    # ---------- 2. Text detection ----------
    tess_cfg = '--oem 3 --psm 6'
    tdata = pytesseract.image_to_data(gray, output_type=Output.DICT, config=tess_cfg)

    text_boxes = []
    for i, txt in enumerate(tdata['text']):
        conf = int(tdata['conf'][i])
        if conf > 25:
            x, y, w, h = (tdata[k][i] for k in ['left', 'top', 'width', 'height'])
            text_boxes.append((x, y, x+w, y+h))
            #roi = img[y:y+h, x:x+w]
            #display(PILImage.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)))
            #print("\n-----------------------------\n")


    # ---------- 3. Diagram / shape detection ----------
    # Mask out text so it won't be re-detected as a shape
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for x1, y1, x2, y2 in text_boxes:
        mask[y1:y2, x1:x2] = 255
    no_text = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(mask))

    edges   = cv2.Canny(no_text, 50, 150)
    edges  = cv2.dilate(edges, None, iterations=1)   # join ANY thin outline
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > 500:                 # ignore speckles
            shape_boxes.append((x, y, x+w, y+h))
            #roi = img[y:y+h, x:x+w]
            #display(PILImage.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)))
            #print("\n-----------------------------\n")

    # Combine all text boxes to get bounding box of the table
    coords = text_boxes + shape_boxes
    if coords:
        xs, ys, xe, ye = zip(*coords)
        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xe), max(ye)

        # Get original image size
        img_height, img_width = img.shape[:2]
        # Compute dynamic paddings based on margins
        left_padding   = min(x_min, 10)
        right_padding  = min(img_width - x_max, 10)
        top_padding    = min(y_min, 10)
        bottom_padding = min(img_height - y_max, 10)

        # Crop table
        cropped_table = img[y_min-top_padding:y_max+bottom_padding, x_min-left_padding:x_max+right_padding]

    # Save the cropped table
    cv2.imwrite(png_path, cropped_table)
    if show_img:
        display(Image(filename=png_path))


def save_latex_table_as_image(table, table_name, output_dir, show_img=False):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tempdir:
        tex_path = os.path.join(tempdir, "document.tex")
        pdf_path = os.path.join(tempdir, "document.pdf")

        # Write LaTeX to .tex file
        with open(tex_path, "w") as f:
            f.write(table)
        # Compile using pdflatex
        assert os.path.exists(tex_path), "TeX file does not exist"
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path],
            cwd=tempdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15
        )
        if result.returncode != 0:
            print("Syntax error in the latex code")
            #print('--------------------------------------------')
            #print(result.stdout.decode()[-400:-40])
            #print('--------------------------------------------')
            raise RuntimeError(f"LaTeX compilation failed:\n{result.stderr.decode()}")

        # Copy the .tex file if compiling succeed
        shutil.copy(tex_path, os.path.join(output_dir, table_name.replace('png', 'tex')))

        # Convert PDF to PNG
        images = convert_from_path(pdf_path, dpi=300)
        png_path = os.path.join(output_dir, table_name)
        images[0].save(png_path)
        print(f"Saved pdf to {png_path}")

        crop_image(png_path, show_img=show_img)



def _strip_code_fences(text: str) -> str:
    # If there are fenced blocks, keep only their contents (often where JSON lives)
    CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    blocks = CODE_FENCE_RE.findall(text)
    return "\n\n".join(blocks) if blocks else text
    
def extract_json_blocks(text):
    """
    Extracts JSON code blocks from the input text and parses them into Python dictionaries.

    Args:
        text (str): The input string potentially containing JSON blocks.

    Returns:
        text (str): A json object in text.
    """
    #If we have a dict of tables
    pattern = r"""
\{\s*                             # Opening brace
\s*"table_1"\s*:\s*".*?"\s*,      # Match "table_1": "<content>"
\s*"table_2"\s*:\s*".*?"\s*,      # Match "table_2": "<content>"
\s*"table_3"\s*:\s*".*?"\s*       # Match "table_3": "<content>"
\}                               # Closing brace
"""

    matches = re.findall(pattern, text, re.DOTALL | re.VERBOSE)
    if matches:
        return matches[-1]
    #If we have a dict of questions answers
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    last_match = None
    for match in matches:
        last_match=match.strip() # We only care about the first match

    if last_match:
        return last_match

    """
    Scan `text` and return the last valid JSON object found as a dict.
    Works even if there's prose before/after, bullets, or code fences.
    """
    # Preprocess: remove code fences (keeps just their content)
    text = _strip_code_fences(text)

    decoder = json.JSONDecoder()
    i = 0
    last_obj = None
    last_str = None

    while i < len(text):
        if text[i] not in "{[":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, idx=i)
            last_obj = obj
            last_str = text[i:end]   # raw substring of the JSON
            i = end
        except json.JSONDecodeError:
            i += 1

    if last_obj is None:
        # We want the text if there was no json found
        # print("No Json found in text:")
        # print(text[:20])
        return text
    else:
        return last_str
    
    return text



def replace_unicode_escape(match):
    try:
        return bytes(match.group(0), 'utf-8').decode('unicode_escape')
    except UnicodeDecodeError:
        return match.group(0)

def decode_llm_latex_output(text):
    # Step 1: Replace double backslashes with single
    unescaped = text.replace('\\\\', '\\')
    # Step 2: Decode any \uXXXX Unicode escape sequences safely
    pattern = r'\\u[0-9a-fA-F]{4}'
    decoded = re.sub(pattern, replace_unicode_escape, unescaped)
    return decoded


def safe_control_escape(seq: str) -> str:
    return {
        r'\n': '\n', r'\t': '\t', r'\r': '\r', r'\b': '\b',
        r'\f': '\f', r'\v': '\v', r'\\"': '"', r"\\'": "'",
    }.get(seq, seq)
def decode_control_sequences(text: str) -> str:
    return re.sub(r"""\\n(?![a-zA-Z])|\\t(?![a-zA-Z])|\\r(?![a-zA-Z])|\\b(?![a-zA-Z])|\\f(?![a-zA-Z])|\\v(?![a-zA-Z])|\\"|\\'""", lambda m: safe_control_escape(m.group(0)), text)


def extract_tables_from_text(text):
    """
    Extracts LaTeX tables from a plain text model response, without relying on JSON parsing.

    Args:
        text (str): The raw string returned by the LLM.

    Returns:
        A dictionary with the 3 keys (table_1, table_2, table_3) containing LaTeX code.
    """
    tables = {}
    pattern = re.compile(r'"(table_\d+)"\s*:\s*"\s*BEGIN_LATEX\s*(.*?)\s*END_LATEX\s*"\s*,?', re.DOTALL)

    for match in pattern.finditer(text):
        key = match.group(1)
        content = match.group(2).strip()
        # Step 1: Decode unicode like \u00e9 → é
        content = decode_llm_latex_output(content) #content.encode().decode('unicode_escape')
        # Step 2: Decode escaped control sequences ONLY if not followed by letters (LaTeX commands)
        tables[key] = decode_control_sequences(content)
    return tables


def update_start_time(time_path):
    now = datetime.now()
    with open(time_path, "r+") as f:
        time_file = json.load(f)
        start_time = time_file['start_time']
        start_time = datetime.fromisoformat(start_time)
        time_diff = now - start_time
        print(f"Time difference: {time_diff}")

        # Check if the difference exceeds 6 hours
        if time_diff > timedelta(hours=6):
            time_file['start_time']= now.isoformat()
            f.seek(0)  # Go back to the start of the file before writing
            json.dump(time_file, f, indent=2)
            f.truncate()
            return True
        return False


def read_Exception(e):
    """
    This function properly reads exceptions thrown by APIs as some of them might be trick to read.
    Handling case by case exceptions might take too long
    """
    msg = ''
    for attr in dir(e):
        if not attr.startswith("_"):
            try:
                value = getattr(e, attr)
                msg += f"{attr}: {value}\n"
            except Exception:
                pass
    return msg
