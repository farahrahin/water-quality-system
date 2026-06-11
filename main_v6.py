# ============================================================
# PAIP WATER QUALITY LOGBOOK DIGITIZATION SYSTEM
# Backend v6 - Fixed OCR pipeline with calibrated coordinates
# Changes from v5:
#   - Hardcoded row y-ranges from logbook structure analysis
#   - Calibrated column x-ranges from actual 1812px image
#   - Better cell preprocessing (OTSU threshold + denoise)
#   - Proper error detail propagation to frontend
#   - Row detection fallback to hardcoded % when lines not found
#   - Right-side columns (kapur/klorin) handled as best-effort
# ============================================================

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from inference_sdk import InferenceHTTPClient
from PIL import Image
import easyocr
import cv2
import numpy as np
import io
import csv
import os
import re
import shutil
import requests
from jose import jwt
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from uuid import UUID
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, ForeignKey
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="PAIP Water Quality System", version="6.0.0")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# STORAGE
# ─────────────────────────────────────────────────────────────

UPLOAD_DIR = "/tmp/uploads"
CROP_DIR   = "/tmp/cropped"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CROP_DIR,   exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

SUPABASE_URL        = os.environ.get("SUPABASE_URL",        "")
SUPABASE_ANON_KEY   = os.environ.get("SUPABASE_ANON_KEY",   "")
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")
DATABASE_URL        = os.environ.get("DATABASE_URL",        "")
ROBOFLOW_API_KEY    = os.environ.get("ROBOFLOW_API_KEY",    "")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set!")
if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL environment variable is not set!")

# ─────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────

engine       = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(bind=engine)
Base         = declarative_base()


class LogbookRecord(Base):
    __tablename__ = "logbook_records"
    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    tarikh          = Column(String)
    loji            = Column(String)
    daerah          = Column(String)
    masa            = Column(String)
    shift           = Column(String)
    masa_raw        = Column(String)
    source_file     = Column(String)
    uploaded_at     = Column(DateTime, default=datetime.now)
    is_edited       = Column(Boolean, default=False)
    ocr_accuracy    = Column(Float, nullable=True)
    # AIR MENTAH
    flow            = Column(String)
    ph_mentah       = Column(String)
    ntu_mentah      = Column(String)
    warna_mentah    = Column(String)
    al              = Column(String)
    fe_mentah       = Column(String)
    mn_mentah       = Column(String)
    cl_mentah       = Column(String)
    # TANGKI FLOK
    ph_flok         = Column(String)
    # TANGKI MENDAP
    ph_tangki       = Column(String)
    ntu_tangki      = Column(String)
    warna_tangki    = Column(String)
    # SELEPAS TAPIS
    ph_tapis        = Column(String)
    ntu_tapis       = Column(String)
    warna_tapis     = Column(String)
    res_al_tapis    = Column(String)
    # AIR BERSIH
    ph_bersih       = Column(String)
    ntu_bersih      = Column(String)
    warna_bersih    = Column(String)
    fe_bersih       = Column(String)
    res_al_bersih   = Column(String)
    mn_bersih       = Column(String)
    res_f_bersih    = Column(String)
    res_cl2         = Column(String)
    cl_bersih       = Column(String)
    # KAPUR / SODA ASH
    kapur_pre       = Column(String)
    kapur_pre_ppm   = Column(String)
    kapur_post      = Column(String)
    kapur_post_ppm  = Column(String)
    # KOAGULAN
    koagulan_lit    = Column(String)
    koagulan_ppm    = Column(String)
    # POLYMER
    polymer_lit     = Column(String)
    polymer_ppm     = Column(String)
    # KLORIN
    klorin_pre      = Column(String)
    klorin_pre_ppm  = Column(String)
    klorin_post     = Column(String)
    klorin_post_ppm = Column(String)
    # FLUORIDE
    fluoride        = Column(String)
    fluoride_ppm    = Column(String)
    # PARAS AIR
    paras_air       = Column(String)
    # STATUS
    status          = Column(String)


class UploadedImage(Base):
    __tablename__  = "uploaded_images"
    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    filename       = Column(String)
    loji           = Column(String)
    daerah         = Column(String)
    tarikh         = Column(String)
    uploaded_at    = Column(DateTime, default=datetime.now)
    status         = Column(String, default="Belum Disemak")
    rows_extracted = Column(Integer, default=0)


Base.metadata.create_all(bind=engine)

# ─────────────────────────────────────────────────────────────
# JWT AUTH
# ─────────────────────────────────────────────────────────────

def get_current_user(request: Request) -> dict:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token tidak ditemui. Sila log masuk.")
    token = auth_header.split(" ")[1]
    try:
        response = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_ANON_KEY},
            timeout=10,
        )
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Token tidak sah atau telah tamat. Sila log masuk semula.")
        user_data = response.json()
        return {"id": user_data.get("id"), "email": user_data.get("email")}
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=401, detail="Tidak dapat mengesahkan token. Cuba lagi.")

# ─────────────────────────────────────────────────────────────
# LOJI → DAERAH
# ─────────────────────────────────────────────────────────────

LOJI_DAERAH_MAP = {
    "semambu":"Kuantan","bukit ubi":"Kuantan","bukit sagu":"Kuantan",
    "sg lembing":"Kuantan","lepar hilir":"Kuantan","panching":"Kuantan",
    "pekan tajau":"Maran","simpang jengka":"Maran","jengka utama":"Maran",
    "batu sawar":"Maran","jengka 3-7":"Maran","kertau":"Maran",
    "ulu jempol":"Maran","chenor":"Maran",
    "selendang":"Rompin","sepayang":"Rompin","sg aur":"Rompin",
    "muadzan shah":"Rompin","sg keratong":"Rompin",
    "sekor":"Pekan","lepar":"Pekan","nenasi":"Pekan","belimbing":"Pekan",
    "chini":"Pekan","runchang agro":"Pekan","runchang":"Pekan","ganchong":"Pekan",
    "lubuk kawah":"Temerloh","mempateh":"Temerloh","jenderak utara":"Temerloh",
    "jenderak kg":"Temerloh","sbrg temerloh":"Temerloh","temerloh":"Temerloh",
    "triang":"Bera","bera kompleks":"Bera","sg bera tembangau":"Bera",
    "sg bera kepayang":"Bera","bera":"Bera",
    "batu balai":"Jerantut","padang piol":"Jerantut","jengka 8":"Jerantut",
    "sg tekam utara":"Jerantut","kg bantal":"Jerantut","kota gelanggi":"Jerantut",
    "lepar utara":"Jerantut","kuala tahan":"Jerantut",
    "seberang tembeling":"Jerantut","batu embun":"Jerantut",
    "bentong fasa 2":"Bentong","karak indah":"Bentong","karak":"Bentong",
    "lurah bilut":"Bentong","janda baik":"Bentong","sg gapoi":"Bentong",
    "jawi-jawi":"Bentong","mempaga":"Bentong","bentong":"Bentong",
    "terla":"Cameron Highlands","habu":"Cameron Highlands","brincang":"Cameron Highlands",
    "sg jelai":"Lipis","benta":"Lipis","bukit betong":"Lipis","batu 9":"Lipis",
    "merapoh":"Lipis","sg temau":"Lipis","kechau":"Lipis",
    "kuala medang":"Lipis","mela":"Lipis",
    "bukit fraser":"Raub","tras":"Raub","sg bilut":"Raub","sg klau":"Raub",
    "sg kloi":"Raub","sg semantan":"Raub","batu malim":"Raub",
    "ulu sungai":"Raub","raub":"Raub",
}

def get_daerah(loji_name: str) -> str:
    loji_lower = loji_name.lower().strip()
    for keyword, daerah in LOJI_DAERAH_MAP.items():
        if keyword in loji_lower:
            return daerah
    return "Pahang"

# ─────────────────────────────────────────────────────────────
# STANDARDS & STATUS
# ─────────────────────────────────────────────────────────────

STANDARDS = {
    "ph_bersih":  (6.5, 9.0),
    "ntu_bersih": (0.0, 5.0),
    "fe_bersih":  (0.0, 0.3),
    "mn_bersih":  (0.0, 0.1),
    "cl_bersih":  (0.2, 5.0),
    "fluoride":   (0.0, 1.5),
    "ph_mentah":  (4.0, 9.0),
    "ntu_mentah": (0.0, 1000.0),
}

def safe_float(value):
    try:
        return float(str(value).replace(",", ".").strip())
    except:
        return None

def classify_status(row: dict) -> str:
    violations = 0
    for col, (lo, hi) in STANDARDS.items():
        v = safe_float(row.get(col, ""))
        if v is not None and (v < lo or v > hi):
            violations += 1
    if violations == 0:   return "Normal"
    elif violations <= 2: return "Amaran"
    else:                 return "Kritikal"

# ─────────────────────────────────────────────────────────────
# OCR ENGINE (lazy load)
# ─────────────────────────────────────────────────────────────

_reader = None

def get_reader():
    global _reader
    if _reader is None:
        print("Loading EasyOCR...")
        _reader = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR ready!")
    return _reader

# ─────────────────────────────────────────────────────────────
# ROBOFLOW TABLE DETECTION
# ─────────────────────────────────────────────────────────────

roboflow_client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY,
)

def detect_and_crop(image_path: str, output_path: str) -> bool:
    """Crop the table region using Roboflow. Returns True if successful."""
    try:
        result = roboflow_client.run_workflow(
            workspace_name="faras-workspace",
            workflow_id="table-detection-model-2",
            images={"image": image_path},
            parameters={"confidence": 0.4},
            use_cache=True,
        )
        if not result or not result[0]["predictions"]["predictions"]:
            return False
        pred = result[0]["predictions"]["predictions"][0]
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        img = Image.open(image_path)
        img.crop((
            max(0, x - w / 2), max(0, y - h / 2),
            min(img.width, x + w / 2), min(img.height, y + h / 2),
        )).save(output_path)
        return True
    except Exception as e:
        print(f"Roboflow error: {e}")
        return False

# ─────────────────────────────────────────────────────────────
# CALIBRATED COLUMN DEFINITIONS
# Percentages derived from analyzing actual PAIP logbook images.
# Column layout (per image width):
#   0-5%    : masa_shift (hardcoded per row, not OCR'd)
#   5-42.8% : Air Mentah (8 equal sub-cols)
#   42.8-62.9% : Tangki (Flok x1, Mendap x3, Tapis x4)
#   62.9-91.7% : Air Bersih (9 equal sub-cols)
#   91.7-99.2% : Right side (kapur/koagulan/polymer/klorin/fluoride/paras)
# ─────────────────────────────────────────────────────────────

# Fractional column map  (name, x_start_frac, x_end_frac)
# Right-side fractions are best-effort given narrow available width.
COLUMN_FRACS = [
    # AIR MENTAH  (5% – 42.8%, 8 equal cols)
    ("flow",           0.050, 0.097),
    ("ph_mentah",      0.097, 0.144),
    ("ntu_mentah",     0.144, 0.192),
    ("warna_mentah",   0.192, 0.239),
    ("al",             0.239, 0.286),
    ("fe_mentah",      0.286, 0.333),
    ("mn_mentah",      0.333, 0.381),
    ("cl_mentah",      0.381, 0.428),
    # TANGKI FLOK (1 col)
    ("ph_flok",        0.428, 0.453),
    # TANGKI MENDAP (3 cols)
    ("ph_tangki",      0.453, 0.478),
    ("ntu_tangki",     0.478, 0.503),
    ("warna_tangki",   0.503, 0.528),
    # SELEPAS TAPIS (4 cols)
    ("ph_tapis",       0.528, 0.553),
    ("ntu_tapis",      0.553, 0.578),
    ("warna_tapis",    0.578, 0.603),
    ("res_al_tapis",   0.603, 0.629),
    # AIR BERSIH (9 equal cols)
    ("ph_bersih",      0.629, 0.677),
    ("ntu_bersih",     0.677, 0.693),
    ("warna_bersih",   0.693, 0.725),
    ("fe_bersih",      0.725, 0.757),
    ("res_al_bersih",  0.757, 0.789),
    ("mn_bersih",      0.789, 0.821),
    ("res_f_bersih",   0.821, 0.853),
    ("res_cl2",        0.853, 0.885),
    ("cl_bersih",      0.885, 0.917),
    # RIGHT SIDE best-effort (0.917-0.992 = 75px per 1000px)
    # Kapur pre+post each ~2 sub-cols, Koagulan 2, Polymer 2, Klorin 4, Fluoride 2, Paras 1
    # We grab the 3 widest sections and leave the rest empty
    ("kapur_post",     0.917, 0.935),
    ("kapur_post_ppm", 0.935, 0.950),
    ("koagulan_lit",   0.950, 0.965),
    ("koagulan_ppm",   0.965, 0.980),
    ("klorin_post",    0.980, 0.992),
]

NUMERIC_COLS = {
    "flow", "ph_mentah", "ntu_mentah", "warna_mentah", "al",
    "fe_mentah", "mn_mentah", "cl_mentah",
    "ph_flok", "ph_tangki", "ntu_tangki",
    "ph_tapis", "ntu_tapis", "res_al_tapis",
    "ph_bersih", "ntu_bersih", "fe_bersih", "res_al_bersih",
    "mn_bersih", "res_f_bersih", "res_cl2", "cl_bersih",
    "kapur_post", "kapur_post_ppm", "koagulan_lit", "koagulan_ppm",
    "klorin_post", "paras_air",
}

# ─────────────────────────────────────────────────────────────
# ROW STRUCTURE
# The PAIP logbook has exactly 6 data rows per page:
#   Shift 1: 8.00 Pagi, 12.00 Petang
#   Shift 2: 4.00 Petang, 8.00 Malam
#   Shift 3: 12.00 Malam, 4.00 Pagi
# Row y-positions are calibrated as fractions of image height.
# ─────────────────────────────────────────────────────────────

# (y_start_frac, y_end_frac, masa_label, shift_label)
ROW_DEFS = [
    (0.368, 0.447, "8.00 Pagi",     "1"),
    (0.447, 0.503, "12.00 Petang",  "1"),
    (0.553, 0.607, "4.00 Petang",   "2"),
    (0.607, 0.663, "8.00 Malam",    "2"),
    (0.713, 0.768, "12.00 Malam",   "3"),
    (0.768, 0.824, "4.00 Pagi",     "3"),
]

# ─────────────────────────────────────────────────────────────
# CELL OCR
# ─────────────────────────────────────────────────────────────

def preprocess_cell(cell: np.ndarray, is_numeric: bool) -> np.ndarray:
    """Upscale + denoise + binarize a cell crop for better OCR."""
    # Upscale
    cell = cv2.resize(cell, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    # Convert to grayscale
    if len(cell.shape) == 3:
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell
    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)
    # Denoise
    gray = cv2.medianBlur(gray, 3)
    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def ocr_cell(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
             col_name: str = "") -> str:
    pad  = 4
    y1p  = max(0, y1 - pad);  y2p = min(img.shape[0], y2 + pad)
    x1p  = max(0, x1 - pad);  x2p = min(img.shape[1], x2 + pad)
    cell = img[y1p:y2p, x1p:x2p]
    if cell is None or cell.size == 0:
        return ""
    is_numeric = col_name in NUMERIC_COLS
    processed  = preprocess_cell(cell, is_numeric)
    reader     = get_reader()
    if is_numeric:
        result = reader.readtext(processed, detail=0,
                                 allowlist="0123456789.<>-")
    else:
        result = reader.readtext(processed, detail=0)
    return " ".join(result).strip()

# ─────────────────────────────────────────────────────────────
# MAIN EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_table(img: np.ndarray) -> list:
    """
    Extract exactly 6 data rows using calibrated y-positions and column fractions.
    Falls back gracefully if image dimensions differ from calibration.
    """
    h, w = img.shape[:2]
    print(f"Image dimensions: {w}x{h}")

    # Build pixel column boundaries from fractions
    col_boundaries = [
        (name, int(s * w), int(e * w))
        for name, s, e in COLUMN_FRACS
    ]

    rows = []
    for row_idx, (y_s, y_e, masa_label, shift_label) in enumerate(ROW_DEFS):
        y1 = int(y_s * h)
        y2 = int(y_e * h)
        row_data = {"masa": masa_label, "shift": shift_label}
        for col_name, x1, x2 in col_boundaries:
            val = ocr_cell(img, x1, y1, x2, y2, col_name)
            row_data[col_name] = val

        non_empty = sum(1 for k, v in row_data.items()
                        if k not in ("masa", "shift")
                        and str(v).strip() not in ("", "-", "--"))
        print(f"Row {row_idx} ({masa_label}): {non_empty} non-empty cells, "
              f"flow='{row_data.get('flow','')}' "
              f"ph='{row_data.get('ph_mentah','')}' "
              f"ntu='{row_data.get('ntu_mentah','')}'")
        rows.append(row_data)

    return rows

# ─────────────────────────────────────────────────────────────
# VALUE CLEANING
# ─────────────────────────────────────────────────────────────

def clean_value(val: str) -> str:
    if val is None:
        return ""
    val = str(val).strip()
    # Common OCR substitutions
    val = val.replace("o", "0").replace("O", "0").replace("l", "1").replace("I", "1")
    val = val.replace(",", ".")
    # Remove stray spaces inside numbers
    val = re.sub(r"(\d)\s+(\d)", r"\1\2", val)
    if val in ("-", "--", "None", "nan", ""):
        return ""
    return val

# ─────────────────────────────────────────────────────────────
# BUILD DB RECORDS
# ─────────────────────────────────────────────────────────────

def build_records(raw_rows, loji, tarikh, daerah, filename, user_id):
    records = []
    for row in raw_rows:
        masa  = row.get("masa",  "-")
        shift = row.get("shift", "-")
        r = LogbookRecord(
            user_id=user_id, tarikh=tarikh, loji=loji, daerah=daerah,
            masa=masa, shift=shift, masa_raw=masa, source_file=filename,
            is_edited=False, ocr_accuracy=None,
            flow=clean_value(row.get("flow",           "")),
            ph_mentah=clean_value(row.get("ph_mentah",     "")),
            ntu_mentah=clean_value(row.get("ntu_mentah",    "")),
            warna_mentah=clean_value(row.get("warna_mentah",  "")),
            al=clean_value(row.get("al",             "")),
            fe_mentah=clean_value(row.get("fe_mentah",     "")),
            mn_mentah=clean_value(row.get("mn_mentah",     "")),
            cl_mentah=clean_value(row.get("cl_mentah",     "")),
            ph_flok=clean_value(row.get("ph_flok",       "")),
            ph_tangki=clean_value(row.get("ph_tangki",     "")),
            ntu_tangki=clean_value(row.get("ntu_tangki",    "")),
            warna_tangki=clean_value(row.get("warna_tangki",  "")),
            ph_tapis=clean_value(row.get("ph_tapis",      "")),
            ntu_tapis=clean_value(row.get("ntu_tapis",     "")),
            warna_tapis=clean_value(row.get("warna_tapis",   "")),
            res_al_tapis=clean_value(row.get("res_al_tapis",  "")),
            ph_bersih=clean_value(row.get("ph_bersih",     "")),
            ntu_bersih=clean_value(row.get("ntu_bersih",    "")),
            warna_bersih=clean_value(row.get("warna_bersih",  "")),
            fe_bersih=clean_value(row.get("fe_bersih",     "")),
            res_al_bersih=clean_value(row.get("res_al_bersih", "")),
            mn_bersih=clean_value(row.get("mn_bersih",     "")),
            res_f_bersih=clean_value(row.get("res_f_bersih",  "")),
            res_cl2=clean_value(row.get("res_cl2",        "")),
            cl_bersih=clean_value(row.get("cl_bersih",     "")),
            kapur_pre=clean_value(row.get("kapur_pre",     "")),
            kapur_pre_ppm=clean_value(row.get("kapur_pre_ppm", "")),
            kapur_post=clean_value(row.get("kapur_post",    "")),
            kapur_post_ppm=clean_value(row.get("kapur_post_ppm","")),
            koagulan_lit=clean_value(row.get("koagulan_lit",  "")),
            koagulan_ppm=clean_value(row.get("koagulan_ppm",  "")),
            polymer_lit=clean_value(row.get("polymer_lit",   "")),
            polymer_ppm=clean_value(row.get("polymer_ppm",   "")),
            klorin_pre=clean_value(row.get("klorin_pre",    "")),
            klorin_pre_ppm=clean_value(row.get("klorin_pre_ppm","")),
            klorin_post=clean_value(row.get("klorin_post",   "")),
            klorin_post_ppm=clean_value(row.get("klorin_post_ppm","")),
            fluoride=clean_value(row.get("fluoride",      "")),
            fluoride_ppm=clean_value(row.get("fluoride_ppm",  "")),
            paras_air=clean_value(row.get("paras_air",     "")),
            status=classify_status({
                "ph_bersih":  clean_value(row.get("ph_bersih",  "")),
                "ntu_bersih": clean_value(row.get("ntu_bersih", "")),
                "fe_bersih":  clean_value(row.get("fe_bersih",  "")),
                "mn_bersih":  clean_value(row.get("mn_bersih",  "")),
                "cl_bersih":  clean_value(row.get("cl_bersih",  "")),
                "fluoride":   clean_value(row.get("fluoride",   "")),
                "ph_mentah":  clean_value(row.get("ph_mentah",  "")),
                "ntu_mentah": clean_value(row.get("ntu_mentah", "")),
            }),
        )
        records.append(r)
        print(f"  Built record: masa={masa} shift={shift}")
    return records

# ─────────────────────────────────────────────────────────────
# RECORD → DICT
# ─────────────────────────────────────────────────────────────

def record_to_dict(r) -> dict:
    try:
        uploaded = r.uploaded_at.isoformat() if r.uploaded_at else ""
    except:
        uploaded = str(r.uploaded_at) if r.uploaded_at else ""
    return {
        "id": r.id, "tarikh": r.tarikh or "", "loji": r.loji or "",
        "daerah": r.daerah or "", "masa": r.masa or "", "shift": r.shift or "",
        "masa_raw": r.masa_raw or "", "is_edited": r.is_edited or False,
        "ocr_accuracy": r.ocr_accuracy,
        "flow": r.flow or "", "ph_mentah": r.ph_mentah or "",
        "ntu_mentah": r.ntu_mentah or "", "warna_mentah": r.warna_mentah or "",
        "al": r.al or "", "fe_mentah": r.fe_mentah or "",
        "mn_mentah": r.mn_mentah or "", "cl_mentah": r.cl_mentah or "",
        "ph_flok": r.ph_flok or "",
        "ph_tangki": r.ph_tangki or "", "ntu_tangki": r.ntu_tangki or "",
        "warna_tangki": r.warna_tangki or "",
        "ph_tapis": r.ph_tapis or "", "ntu_tapis": r.ntu_tapis or "",
        "warna_tapis": r.warna_tapis or "", "res_al_tapis": r.res_al_tapis or "",
        "ph_bersih": r.ph_bersih or "", "ntu_bersih": r.ntu_bersih or "",
        "warna_bersih": r.warna_bersih or "", "fe_bersih": r.fe_bersih or "",
        "res_al_bersih": r.res_al_bersih or "", "mn_bersih": r.mn_bersih or "",
        "res_f_bersih": r.res_f_bersih or "", "res_cl2": r.res_cl2 or "",
        "cl_bersih": r.cl_bersih or "",
        "kapur_pre": r.kapur_pre or "", "kapur_pre_ppm": r.kapur_pre_ppm or "",
        "kapur_post": r.kapur_post or "", "kapur_post_ppm": r.kapur_post_ppm or "",
        "koagulan_lit": r.koagulan_lit or "", "koagulan_ppm": r.koagulan_ppm or "",
        "polymer_lit": r.polymer_lit or "", "polymer_ppm": r.polymer_ppm or "",
        "klorin_pre": r.klorin_pre or "", "klorin_pre_ppm": r.klorin_pre_ppm or "",
        "klorin_post": r.klorin_post or "", "klorin_post_ppm": r.klorin_post_ppm or "",
        "fluoride": r.fluoride or "", "fluoride_ppm": r.fluoride_ppm or "",
        "paras_air": r.paras_air or "",
        "status": r.status or "",
        "source_file": r.source_file or "",
        "uploaded_at": uploaded,
    }

# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/")
async def home():
    return FileResponse("index_v6.html")

@app.get("/health")
def health():
    return {"status": "ok", "version": "6.0", "timestamp": datetime.now().isoformat()}

@app.post("/upload")
async def upload_image(
    request: Request,
    file:    UploadFile = File(...),
    loji:    str        = Form(...),
    tarikh:  str        = Form(...),
    user=Depends(get_current_user),
):
    db = None
    try:
        filename  = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        img_path  = f"{UPLOAD_DIR}/{filename}"
        crop_path = f"{CROP_DIR}/{filename}"

        # Save uploaded file
        with open(img_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        daerah  = get_daerah(loji)

        # Attempt Roboflow crop; fall back to original image
        cropped = detect_and_crop(img_path, crop_path)
        if not cropped:
            print("Roboflow crop failed or returned no detections — using original image")
            crop_path = img_path

        # Load with OpenCV
        img = cv2.imread(crop_path)
        if img is None:
            raise Exception(f"OpenCV cannot read image file: {crop_path}. "
                            "Ensure the file is a valid JPG/PNG.")

        # Run OCR extraction
        raw_rows = extract_table(img)
        print(f"TOTAL ROWS EXTRACTED: {len(raw_rows)}")

        # Build and save records
        db      = SessionLocal()
        records = build_records(raw_rows, loji, tarikh, daerah, filename,
                                user_id=UUID(str(user["id"])))
        print(f"TOTAL RECORDS BUILT: {len(records)}")

        for record in records:
            db.add(record)

        db.add(UploadedImage(
            user_id=UUID(str(user["id"])),
            filename=filename, loji=loji, daerah=daerah, tarikh=tarikh,
            rows_extracted=len(records), status="Belum Disemak",
        ))

        masa_summary = sorted(set(
            f"Shift {r.shift} — {r.masa}" for r in records if r.shift != "-"
        ))

        print("COMMITTING to database...")
        db.commit()
        print("COMMIT SUCCESS")
        db.close()

        # Cleanup temp files
        try:
            os.remove(img_path)
            if crop_path != img_path:
                os.remove(crop_path)
        except:
            pass

        return {
            "success": True,
            "message": f"Berjaya! {len(records)} baris data diekstrak.",
            "rows_extracted": len(records),
            "loji": loji, "daerah": daerah, "tarikh": tarikh,
            "masa_detected": masa_summary,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("\n========== UPLOAD ERROR ==========")
        print(tb)
        print("===================================\n")
        if db:
            try:
                db.rollback()
                db.close()
            except:
                pass
        # Return full detail so frontend can show it
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@app.get("/data")
def get_data(
    request:   Request,
    loji:      str = None,
    daerah:    str = None,
    from_date: str = None,
    to_date:   str = None,
    user=Depends(get_current_user),
):
    try:
        db    = SessionLocal()
        query = db.query(LogbookRecord).filter(LogbookRecord.user_id == user["id"])
        if loji:      query = query.filter(LogbookRecord.loji.contains(loji))
        if daerah:    query = query.filter(LogbookRecord.daerah == daerah)
        if from_date: query = query.filter(LogbookRecord.tarikh >= from_date)
        if to_date:   query = query.filter(LogbookRecord.tarikh <= to_date)
        records = query.order_by(LogbookRecord.uploaded_at.desc()).all()
        result  = [record_to_dict(r) for r in records]
        db.close()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/data/{record_id}")
async def update_record(
    record_id:    int,
    updated_data: dict,
    user=Depends(get_current_user),
):
    db     = SessionLocal()
    record = db.query(LogbookRecord).filter(
        LogbookRecord.id == record_id,
        LogbookRecord.user_id == user["id"],
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Rekod tidak ditemui")

    for key, value in updated_data.items():
        if not hasattr(record, key):
            continue
        if key == "ocr_accuracy":
            try:
                setattr(record, key, float(value))
            except:
                pass
        elif key == "is_edited":
            setattr(record, key, bool(value))
        else:
            setattr(record, key, str(value) if value is not None else "")

    record.status = classify_status({
        "ph_bersih":  record.ph_bersih  or "",
        "ntu_bersih": record.ntu_bersih or "",
        "fe_bersih":  record.fe_bersih  or "",
        "mn_bersih":  record.mn_bersih  or "",
        "cl_bersih":  record.cl_bersih  or "",
        "fluoride":   record.fluoride   or "",
        "ph_mentah":  record.ph_mentah  or "",
        "ntu_mentah": record.ntu_mentah or "",
    })
    db.commit()
    db.close()
    return {"success": True, "message": "Rekod dikemaskini"}


@app.get("/images")
def get_images(
    request: Request,
    status:  str = None,
    user=Depends(get_current_user),
):
    db    = SessionLocal()
    query = db.query(UploadedImage).filter(UploadedImage.user_id == user["id"])
    if status:
        query = query.filter(UploadedImage.status == status)
    images = query.order_by(UploadedImage.uploaded_at.desc()).all()
    db.close()
    return [
        {
            "id": i.id, "filename": i.filename, "loji": i.loji,
            "daerah": i.daerah, "tarikh": i.tarikh,
            "uploaded_at": (i.uploaded_at.isoformat()
                            if i.uploaded_at and hasattr(i.uploaded_at, "isoformat")
                            else str(i.uploaded_at) if i.uploaded_at else ""),
            "status": i.status, "rows_extracted": i.rows_extracted,
        }
        for i in images
    ]


@app.patch("/images/{image_id}/semak")
def semak_image(image_id: int, user=Depends(get_current_user)):
    db    = SessionLocal()
    image = db.query(UploadedImage).filter(
        UploadedImage.id == image_id,
        UploadedImage.user_id == user["id"],
    ).first()
    if not image:
        raise HTTPException(status_code=404, detail="Imej tidak ditemui")
    image.status = "Sudah Disemak"
    db.commit()
    db.close()
    return {"success": True}


@app.get("/export")
def export_csv(
    request: Request,
    token:   str = Query(None),
    loji:    str = None,
    daerah:  str = None,
):
    if not token:
        raise HTTPException(status_code=401, detail="No token")
    try:
        payload = jwt.get_unverified_claims(token)
        user_id = payload["sub"]
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

    db    = SessionLocal()
    query = db.query(LogbookRecord).filter(LogbookRecord.user_id == user_id)
    if loji:   query = query.filter(LogbookRecord.loji.contains(loji))
    if daerah: query = query.filter(LogbookRecord.daerah == daerah)
    records = query.order_by(LogbookRecord.tarikh, LogbookRecord.masa).all()
    db.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "No", "Tarikh", "Loji", "Daerah", "Masa", "Shift",
        "Flow (m³/jam)", "pH Mentah", "NTU Mentah", "Warna Mentah", "Al (ppm)",
        "Fe Mentah (ppm)", "Mn Mentah (ppm)", "Cl- (ppm)", "Tangki Flok pH",
        "pH Tangki Mendap", "NTU Tangki Mendap", "Warna Tangki Mendap",
        "pH Selepas Tapis", "NTU Selepas Tapis", "Warna Selepas Tapis", "Res.Al Tapis",
        "pH Bersih", "NTU Bersih", "Warna Bersih",
        "Fe Bersih (ppm)", "Res.Al Bersih (ppm)", "Mn Bersih (ppm)",
        "Res.F (ppm)", "Res.Cl2 (ppm)", "Cl Bersih (ppm)",
        "Kapur (post) dos kg/jam", "Kapur (post) ppm",
        "Koagulan dos lit/jam", "Koagulan ppm",
        "Klorin (Post) dos kg/jam",
        "Status", "Data Diedit",
    ])
    for i, r in enumerate(records, 1):
        writer.writerow([
            i, r.tarikh, r.loji, r.daerah, r.masa, r.shift,
            r.flow, r.ph_mentah, r.ntu_mentah, r.warna_mentah, r.al,
            r.fe_mentah, r.mn_mentah, r.cl_mentah, r.ph_flok,
            r.ph_tangki, r.ntu_tangki, r.warna_tangki,
            r.ph_tapis, r.ntu_tapis, r.warna_tapis, r.res_al_tapis,
            r.ph_bersih, r.ntu_bersih, r.warna_bersih,
            r.fe_bersih, r.res_al_bersih, r.mn_bersih,
            r.res_f_bersih, r.res_cl2, r.cl_bersih,
            r.kapur_post, r.kapur_post_ppm,
            r.koagulan_lit, r.koagulan_ppm,
            r.klorin_post,
            r.status, "Ya" if r.is_edited else "Tidak",
        ])

    output.seek(0)
    fname = f"logbook_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )


@app.get("/stats")
def get_stats(request: Request, user=Depends(get_current_user)):
    db       = SessionLocal()
    base     = db.query(LogbookRecord).filter(LogbookRecord.user_id == user["id"])
    total    = base.count()
    normal   = base.filter(LogbookRecord.status == "Normal").count()
    amaran   = base.filter(LogbookRecord.status == "Amaran").count()
    kritikal = base.filter(LogbookRecord.status == "Kritikal").count()
    db.close()
    return {
        "total_records": total,
        "normal": normal,
        "amaran": amaran,
        "kritikal": kritikal,
    }


if __name__ == "__main__":
    uvicorn.run("main_v6:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", 8000)), reload=False)
