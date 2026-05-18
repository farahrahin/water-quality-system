# ============================================================
# PAIP WATER QUALITY LOGBOOK DIGITIZATION SYSTEM
# Backend v5 - Supabase PostgreSQL + JWT Authentication
# ============================================================
# SETUP:
# pip install fastapi uvicorn python-multipart sqlalchemy psycopg2-binary
#             easyocr opencv-python pillow inference-sdk PyJWT requests
# python main.py
# ============================================================

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, text
from sqlalchemy.orm import declarative_base
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

app = FastAPI(title="PAIP Water Quality System", version="5.0.0")
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("cropped", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("cropped", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SUPABASE CONFIG
# ─────────────────────────────────────────────────────────────

SUPABASE_URL      = "https://eeczzlnboymjuxcdfxze.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVlY3p6bG5ib3ltanV4Y2RmeHplIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg3NTI2NTksImV4cCI6MjA5NDMyODY1OX0.cxnbjL-6zFLUe_EPe-Ruc27KZx_SHJkyflM1mnJBD0g"
SUPABASE_JWT_SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"

# ─────────────────────────────────────────────────────────────
# DATABASE - Supabase PostgreSQL
# ─────────────────────────────────────────────────────────────

DATABASE_URL = "postgresql://postgres.eeczzlnboymjuxcdfxze:DSPpaip2026!@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
)
SessionLocal = sessionmaker(bind=engine)
Base         = declarative_base()


class LogbookRecord(Base):
    __tablename__ = "logbook_records"
    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    # Metadata
    tarikh         = Column(String)
    loji           = Column(String)
    daerah         = Column(String)
    masa           = Column(String)   # auto-detected from OCR
    shift          = Column(String)   # auto-detected from masa
    masa_raw       = Column(String)   # raw OCR text from masa column
    source_file    = Column(String)
    uploaded_at    = Column(DateTime, default=datetime.now)
    is_edited      = Column(Boolean, default=False)
    ocr_accuracy   = Column(Float, nullable=True)

    # ── AIR MENTAH (8 cols) ──
    flow           = Column(String)
    ph_mentah      = Column(String)
    ntu_mentah     = Column(String)
    warna_mentah   = Column(String)
    al             = Column(String)
    fe_mentah      = Column(String)
    mn_mentah      = Column(String)
    cl_mentah      = Column(String)

    # ── TANGKI FLOK (pH only) ──
    ph_flok        = Column(String)

    # ── TANGKI MENDAP (3 cols) ──
    ph_tangki      = Column(String)
    ntu_tangki     = Column(String)
    warna_tangki   = Column(String)

    # ── SELEPAS TAPIS (4 cols) ──
    ph_tapis       = Column(String)
    ntu_tapis      = Column(String)
    warna_tapis    = Column(String)
    res_al_tapis   = Column(String)

    # ── AIR BERSIH (9 cols) ──
    ph_bersih      = Column(String)
    ntu_bersih     = Column(String)
    warna_bersih   = Column(String)
    fe_bersih      = Column(String)
    res_al_bersih  = Column(String)
    mn_bersih      = Column(String)
    res_f_bersih   = Column(String)
    res_cl2        = Column(String)
    cl_bersih      = Column(String)

    # ── KAPUR / SODA ASH (4 cols) ──
    kapur_pre      = Column(String)
    kapur_pre_ppm  = Column(String)
    kapur_post     = Column(String)
    kapur_post_ppm = Column(String)

    # ── KOAGULAN (2 cols) ──
    koagulan_lit   = Column(String)
    koagulan_ppm   = Column(String)

    # ── POLYMER (2 cols) ──
    polymer_lit    = Column(String)
    polymer_ppm    = Column(String)

    # ── KLORIN (4 cols) ──
    klorin_pre     = Column(String)
    klorin_pre_ppm = Column(String)
    klorin_post    = Column(String)
    klorin_post_ppm = Column(String)

    # ── FLUORIDE (2 cols) ──
    fluoride       = Column(String)
    fluoride_ppm   = Column(String)

    # ── PARAS AIR ──
    paras_air      = Column(String)

    # ── STATUS ──
    status         = Column(String)


class UploadedImage(Base):
    __tablename__   = "uploaded_images"
    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    filename        = Column(String)
    loji            = Column(String)
    daerah          = Column(String)
    tarikh          = Column(String)
    uploaded_at     = Column(DateTime, default=datetime.now)
    status          = Column(String, default="Belum Disemak")
    rows_extracted  = Column(Integer, default=0)


Base.metadata.create_all(bind=engine)

# ─────────────────────────────────────────────────────────────
# JWT AUTH - Verify Supabase token
# ─────────────────────────────────────────────────────────────

def get_current_user(request: Request) -> dict:
    """
    Verify Supabase JWT token from Authorization header.
    Returns user dict with 'sub' (user UUID) and 'email'.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token tidak ditemui. Sila log masuk.")

    token = auth_header.split(" ")[1]

    try:
        # Verify token with Supabase
        response = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": SUPABASE_ANON_KEY,
            },
            timeout=10
        )
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Token tidak sah atau telah tamat. Sila log masuk semula.")

        user_data = response.json()
        return {
            "id":    user_data.get("id"),
            "email": user_data.get("email"),
        }

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
# MASA / SHIFT AUTO-DETECTION
# ─────────────────────────────────────────────────────────────

def parse_masa_shift(raw_text: str) -> dict:

    import re

    text = str(raw_text).lower().strip()

    # OCR correction
    text = text.replace('o','0')
    text = text.replace('l','1')

    # buang extra spacing
    text = re.sub(r'\s+','',text)

    print("DEBUG OCR:", text)

    hour=None

    m = re.search(r'(\d{1,2})[.:]?(\d{2})?', text)

    if m:
        hour=int(m.group(1))


    is_pg = any(k in text for k in [
        'pg','pagi'
    ])

    is_tgh = any(k in text for k in [
        'ptg12',
        'tgh',
        'tengah',
        'tghari'
    ])

    is_ptg = any(k in text for k in [
        'ptg',
        'petang'
    ])

    is_mlm = any(k in text for k in [
        'mlm',
        'm1m',
        'malam'
    ])


    # SHIFT 1
    if hour==8 and is_pg:
        return {"masa":"8.00 Pagi","shift":"1"}

    if hour==12 and is_ptg:
        return {"masa":"12.00 Petang","shift":"1"}


    # SHIFT 2
    if hour==4 and is_ptg:
        return {"masa":"4.00 Petang","shift":"2"}

    if hour==8 and is_mlm:
        return {"masa":"8.00 Malam","shift":"2"}


    # SHIFT 3
    if hour==12 and is_mlm:
        return {"masa":"12.00 Malam","shift":"3"}

    if hour==4 and is_pg:
        return {"masa":"4.00 Pagi","shift":"3"}


    # fallback — jangan paksa masa salah
    if hour == 8:
        return {"masa": raw_text, "shift":"-"}

    if hour == 12:
        return {"masa": raw_text, "shift":"-"}

    if hour == 4:
        return {"masa": raw_text, "shift":"-"}

    return {
        "masa":"-",
        "shift":"-"
    }

# ─────────────────────────────────────────────────────────────
# OCR PIPELINE
# ─────────────────────────────────────────────────────────────

print("Loading EasyOCR...")

reader = None

def get_reader():
    global reader

    if reader is None:

        print("Loading EasyOCR...")

        reader = easyocr.Reader(
            ['en'],
            gpu=False
        )

        print("EasyOCR Ready")

    return reader

roboflow_client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="agbNxAlXR4nzvN871ZAT"
)

NUMERIC_COLS = {
    'flow', 'ph_mentah', 'ntu_mentah', 'warna_mentah', 'al',
    'fe_mentah', 'mn_mentah', 'cl_mentah',
    'ph_flok', 'ph_tangki', 'ntu_tangki', 'warna_tangki',
    'ph_tapis', 'ntu_tapis', 'res_al_tapis',
    'ph_bersih', 'ntu_bersih', 'fe_bersih', 'res_al_bersih',
    'mn_bersih', 'res_f_bersih', 'res_cl2', 'cl_bersih', 'paras_air'
}

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
        if v is not None:
            if v < lo or v > hi:
                violations += 1
    if violations == 0:   return "Normal"
    elif violations <= 2: return "Amaran"
    else:                 return "Kritikal"

def detect_and_crop(image_path: str, output_path: str) -> bool:
    try:
        result = roboflow_client.run_workflow(
            workspace_name="faras-workspace",
            workflow_id="table-detection-model-2",
            images={"image": image_path},
            parameters={"confidence": 0.4},
            use_cache=False
        )
        if not result or not result[0]['predictions']['predictions']:
            return False
        pred = result[0]['predictions']['predictions'][0]
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        img  = Image.open(image_path)
        img.crop((
            max(0, x-w/2), max(0, y-h/2),
            min(img.width, x+w/2), min(img.height, y+h/2)
        )).save(output_path)
        return True
    except Exception as e:
        print(f"Roboflow error: {e}")
        return False

def get_row_boundaries(img: np.ndarray) -> list:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )
    enhanced = clahe.apply(gray)

    thresh = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        10
    )

    # kurang agresif
    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (img.shape[1]//5,1)
    )

    h_lines = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        h_kernel,
        iterations=2
    )

    row_sums = np.sum(h_lines, axis=1)

    # turunkan threshold
    line_ys = np.where(
        row_sums > img.shape[1]*0.25
    )[0]

    boundaries=[]

    if len(line_ys)>0:

        cluster=[line_ys[0]]
        prev=line_ys[0]

        for y in line_ys[1:]:

            if y-prev<=8:
                cluster.append(y)

            else:
                boundaries.append(
                    int(np.mean(cluster))
                )

                cluster=[y]

            prev=y

        boundaries.append(
            int(np.mean(cluster))
        )

    print("BOUNDARIES:",boundaries)

    return boundaries

def get_column_boundaries(img_width: int):

    cols = [
        ("masa_shift",     0.000, 0.055),

        # AIR MENTAH
        ("flow",           0.055, 0.095),
        ("ph_mentah",      0.095, 0.130),
        ("ntu_mentah",     0.130, 0.165),
        ("warna_mentah",   0.165, 0.205),
        ("al",             0.205, 0.240),
        ("fe_mentah",      0.240, 0.275),
        ("mn_mentah",      0.275, 0.310),
        ("cl_mentah",      0.310, 0.345),

        # TANGKI FLOK
        ("ph_flok",        0.345, 0.380),

        # TANGKI MENDAP
        ("ph_tangki",      0.380, 0.415),
        ("ntu_tangki",     0.415, 0.450),
        ("warna_tangki",   0.450, 0.490),

        # SELEPAS TAPIS
        ("ph_tapis",       0.490, 0.525),
        ("ntu_tapis",      0.525, 0.560),
        ("warna_tapis",    0.560, 0.600),
        ("res_al_tapis",   0.600, 0.640),

        # AIR BERSIH
        ("ph_bersih",      0.640, 0.675),
        ("ntu_bersih",     0.675, 0.710),
        ("warna_bersih",   0.710, 0.750),
        ("fe_bersih",      0.750, 0.785),
        ("res_al_bersih",  0.785, 0.820),
        ("mn_bersih",      0.820, 0.855),
        ("res_f_bersih",   0.855, 0.890),
        ("res_cl2",        0.890, 0.925),
        ("cl_bersih",      0.925, 0.960),

        # KAPUR
        ("kapur_pre",      0.960, 1.000),
        ("kapur_pre_ppm",  1.000, 1.035),
        ("kapur_post",     1.035, 1.075),
        ("kapur_post_ppm", 1.075, 1.110),

        # KOAGULAN
        ("koagulan_lit",   1.110, 1.145),
        ("koagulan_ppm",   1.145, 1.180),

        # POLYMER
        ("polymer_lit",    1.180, 1.215),
        ("polymer_ppm",    1.215, 1.250),

        # KLORIN
        ("klorin_pre",     1.250, 1.290),
        ("klorin_pre_ppm", 1.290, 1.325),
        ("klorin_post",    1.325, 1.365),
        ("klorin_post_ppm",1.365, 1.400),

        # FLUORIDE
        ("fluoride",       1.400, 1.435),
        ("fluoride_ppm",   1.435, 1.470),

        # PARAS AIR
        ("paras_air",      1.470, 1.520),
    ]

    # normalize supaya max = 1.0
    max_x = max(e for _,_,e in cols)

    cols = [
        (name, s/max_x, e/max_x)
        for name,s,e in cols
    ]

    return [
        (name,
         int(s*img_width),
         int(e*img_width))
        for name,s,e in cols
    ]

def ocr_cell(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    col_name: str=""
) -> str:

    pad=3

    y1p=max(0,y1-pad)
    y2p=min(img.shape[0],y2+pad)

    x1p=max(0,x1-pad)
    x2p=min(img.shape[1],x2+pad)

    cell=img[y1p:y2p,x1p:x2p]

    if cell is None or cell.size==0:
        return ""

    cell_big=cv2.resize(
        cell,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    cell_clean=cv2.medianBlur(
        cell_big,
        3
    )

    ocr_reader = get_reader()

    if col_name in NUMERIC_COLS:

        result=ocr_reader.readtext(
            cell_clean,
            detail=0,
            allowlist='0123456789.<>-'
        )

    else:

        result=ocr_reader.readtext(
            cell_clean,
            detail=0
        )

    return " ".join(result).strip()

    except Exception as e:

        print("OCR ERROR:", e)

        return ""
    
def extract_table(img: np.ndarray,
                  row_boundaries: list,
                  col_boundaries: list,
                  header_skip_pct: float = 0.28,
                  min_row_height: int = 20) -> list:

    img_h = img.shape[0]

    # skip header atas
    data_start_y = int(img_h * header_skip_pct)

    # ambik row bawah header je
    data_rows = [y for y in row_boundaries if y > data_start_y]

    all_rows = []

    for i in range(len(data_rows)-1):

        y1 = data_rows[i]
        y2 = data_rows[i+1]

        # buang row terlalu kecil
        if (y2-y1) < min_row_height:
            continue

        row_data = {}

        # OCR setiap column
        for col_name, x1, x2 in col_boundaries:
            row_data[col_name] = ocr_cell(
                img,
                x1,
                y1,
                x2,
                y2,
                col_name
            )

        # check row kosong
        non_empty = sum(
            1 for v in row_data.values()
            if str(v).strip() not in ["", "-", "--"]
        )

        # kalau kurang 3 field isi -> ignore
        if non_empty < 3:
            continue

        all_rows.append(row_data)

        print(
            f"Row {len(all_rows)} "
            f"flow='{row_data.get('flow','')}'"
        )


    # ===========================
    # AUTO ASSIGN MASA + SHIFT
    # ===========================

    masa_order = [
        "8.00 pg",
        "12.00 ptg",
        "4.00 ptg",
        "8.00 mlm",
        "12.00 mlm",
        "4.00 pg"
    ]

    shift_order = [
        "1",
        "1",
        "2",
        "2",
        "3",
        "3"
    ]


    # kalau detect lebih 6 row
    # ambik 6 paling atas je
    all_rows = all_rows[:6]


    for idx,row in enumerate(all_rows):

        row["masa"] = masa_order[idx]

        row["shift"] = shift_order[idx]

        # overwrite OCR masa rosak
        row["masa_shift"] = masa_order[idx]

        print(
            f"→ masa='{row['masa']}' "
            f"shift='{row['shift']}'"
        )

    return all_rows

def clean_value(val):

    if val is None:
        return ""

    val = str(val).strip()

    # OCR fixes
    val = val.replace("o","0")
    val = val.replace("O","0")
    val = val.replace("l","1")

    # junk values
    if val in ["-","--","None","nan"]:
        return ""

    return val

def normalize_row_keys(row):

    clean={}

    for k,v in row.items():

        new_key = k.split("__")[0]

        clean[new_key]=v

    return clean

def build_records(raw_rows, loji, tarikh, daerah, filename, user_id):
    records = []

    for row in raw_rows:

        row = normalize_row_keys(row)

        print("CLEANED:", row)

        masa_raw = row.get("masa_shift", "")
        masa_info = parse_masa_shift(masa_raw)

        masa = masa_info["masa"]
        shift = masa_info["shift"]

        r = LogbookRecord(
            user_id=user_id,  
            tarikh=tarikh,
            loji=loji,
            daerah=daerah,
            masa=masa,
            shift=shift,
            masa_raw=masa_raw,
            source_file=filename,
            is_edited=False,
            ocr_accuracy=None,

            flow=clean_value(row.get("flow", "")),
            ph_mentah=clean_value(row.get("ph_mentah", "")),
            ntu_mentah=clean_value(row.get("ntu_mentah", "")),
            warna_mentah=clean_value(row.get("warna_mentah", "")),
            al=clean_value(row.get("al", "")),
            fe_mentah=clean_value(row.get("fe_mentah", "")),
            mn_mentah=clean_value(row.get("mn_mentah", "")),
            cl_mentah=clean_value(row.get("cl_mentah", "")),
            ph_flok=clean_value(row.get("ph_flok", "")),
            ph_tangki=clean_value(row.get("ph_tangki", "")),
            ntu_tangki=clean_value(row.get("ntu_tangki", "")),
            warna_tangki=clean_value(row.get("warna_tangki", "")),
            ph_tapis=clean_value(row.get("ph_tapis", "")),
            ntu_tapis=clean_value(row.get("ntu_tapis", "")),
            warna_tapis=clean_value(row.get("warna_tapis", "")),
            res_al_tapis=clean_value(row.get("res_al_tapis", "")),
            ph_bersih=clean_value(row.get("ph_bersih", "")),
            ntu_bersih=clean_value(row.get("ntu_bersih", "")),
            warna_bersih=clean_value(row.get("warna_bersih", "")),
            fe_bersih=clean_value(row.get("fe_bersih", "")),
            res_al_bersih=clean_value(row.get("res_al_bersih", "")),
            mn_bersih=clean_value(row.get("mn_bersih", "")),
            res_f_bersih=clean_value(row.get("res_f_bersih", "")),
            res_cl2=clean_value(row.get("res_cl2", "")),
            cl_bersih=clean_value(row.get("cl_bersih", "")),
            kapur_pre=clean_value(row.get("kapur_pre", "")),
            kapur_pre_ppm=clean_value(row.get("kapur_pre_ppm", "")),
            kapur_post=clean_value(row.get("kapur_post", "")),
            kapur_post_ppm=clean_value(row.get("kapur_post_ppm", "")),
            koagulan_lit=clean_value(row.get("koagulan_lit", "")),
            koagulan_ppm=clean_value(row.get("koagulan_ppm", "")),
            polymer_lit=clean_value(row.get("polymer_lit", "")),
            polymer_ppm=clean_value(row.get("polymer_ppm", "")),
            klorin_pre=clean_value(row.get("klorin_pre", "")),
            klorin_pre_ppm=clean_value(row.get("klorin_pre_ppm", "")),
            klorin_post=clean_value(row.get("klorin_post", "")),
            klorin_post_ppm=clean_value(row.get("klorin_post_ppm", "")),
            fluoride=clean_value(row.get("fluoride", "")),
            fluoride_ppm=clean_value(row.get("fluoride_ppm", "")),
            paras_air=clean_value(row.get("paras_air", "")),

            status=classify_status({
                "ph_bersih": clean_value(row.get("ph_bersih","")),
                "ntu_bersih": clean_value(row.get("ntu_bersih","")),
                "fe_bersih": clean_value(row.get("fe_bersih","")),
                "mn_bersih": clean_value(row.get("mn_bersih","")),
                "cl_bersih": clean_value(row.get("cl_bersih","")),
                "fluoride": clean_value(row.get("fluoride","")),
                "ph_mentah": clean_value(row.get("ph_mentah","")),
                "ntu_mentah": clean_value(row.get("ntu_mentah",""))
            })
        )

        records.append(r)

        print(f"→ masa='{masa}' shift='{shift}'")

    return records

# ─────────────────────────────────────────────────────────────
# HELPER: record to dict
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
        "flow": r.flow or "", "ph_mentah": r.ph_mentah or "", "ntu_mentah": r.ntu_mentah or "",
        "warna_mentah": r.warna_mentah or "", "al": r.al or "",
        "fe_mentah": r.fe_mentah or "", "mn_mentah": r.mn_mentah or "",
        "cl_mentah": r.cl_mentah or "", 
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
    return FileResponse("index_v5.html")

@app.get("/health")
def health():
    return {"status":"ok","version":"5.0","timestamp":datetime.now().isoformat()}

@app.post("/upload")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    loji: str = Form(...),
    tarikh: str = Form(...),
    user=Depends(get_current_user),
):
    db = None

    try:
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        img_path = f"uploads/{filename}"
        crop_path = f"cropped/{filename}"

        with open(img_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        daerah = get_daerah(loji)

        cropped = detect_and_crop(img_path, crop_path)
        if not cropped:
            crop_path = img_path

        img = cv2.imread(crop_path)

        if img is None:
            raise Exception(f"Cannot read image: {crop_path}")

        row_boundaries = get_row_boundaries(img)
        col_boundaries = get_column_boundaries(img.shape[1])
        raw_rows = extract_table(
            img,
            row_boundaries,
            col_boundaries
        )

        print("TOTAL ROWS:", len(raw_rows))

        db = SessionLocal()

        records = build_records(
            raw_rows,
            loji,
            tarikh,
            daerah,
            filename,
            user_id=UUID(str(user["id"]))
        )

        print("TOTAL RECORDS:", len(records))

        for record in records:
            db.add(record)

        db.add(
            UploadedImage(
                user_id=UUID(str(user["id"])),
                filename=filename,
                loji=loji,
                daerah=daerah,
                tarikh=tarikh,
                rows_extracted=len(records),
                status="Belum Disemak"
            )
        )

        masa_summary = list(
            set(
                f"Shift {r.shift} — {r.masa}"
                for r in records
                if r.shift != "-"
            )
        )

        print("COMMITTING DATABASE...")

        db.commit()

        print("COMMIT SUCCESS")

        db.close()

        return {
            "success": True,
            "message": f"Berjaya! {len(records)} baris data diekstrak.",
            "rows_extracted": len(records),
            "loji": loji,
            "daerah": daerah,
            "tarikh": tarikh,
            "masa_detected": masa_summary,
        }

    except Exception as e:

        import traceback

        print("\n========== REAL ERROR ==========")
        traceback.print_exc()
        print("TYPE:", type(e))
        print("ERROR:", repr(e))
        print("================================\n")

        if db:
            try:
                db.rollback()
                db.close()
            except:
                pass

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

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
        query = db.query(LogbookRecord).filter(
            LogbookRecord.user_id == user["id"]
        )
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
        LogbookRecord.user_id == user["id"]
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Rekod tidak ditemui")

    for key, value in updated_data.items():
        if not hasattr(record, key): continue
        if key == "ocr_accuracy":
            try: setattr(record, key, float(value))
            except: pass
        elif key == "is_edited":
            setattr(record, key, bool(value))
        else:
            setattr(record, key, str(value) if value is not None else "")

    if "masa" in updated_data:
        info = parse_masa_shift(updated_data["masa"])
        record.shift = info["shift"]

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
            "id":i.id,"filename":i.filename,"loji":i.loji,
            "daerah":i.daerah,"tarikh":i.tarikh,
            "uploaded_at": i.uploaded_at.isoformat() if i.uploaded_at and hasattr(i.uploaded_at, 'isoformat') else str(i.uploaded_at) if i.uploaded_at else "",
            "status":i.status,"rows_extracted":i.rows_extracted,
        }
        for i in images
    ]


@app.patch("/images/{image_id}/semak")
def semak_image(
    image_id: int,
    user=Depends(get_current_user),
):
    db    = SessionLocal()
    image = db.query(UploadedImage).filter(
        UploadedImage.id == image_id,
        UploadedImage.user_id == user["id"]
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
    token: str = Query(None),
    loji: str = None,
    daerah: str = None
):
    if not token:
        raise HTTPException(
            status_code=401,
            detail="No token"
        )

    try:
        payload = jwt.get_unverified_claims(token)

        user_id = payload["sub"]

        print("USER ID:", user_id)

    except Exception as e:
        print(e)

        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )
    db    = SessionLocal()
    query = db.query(LogbookRecord).filter(LogbookRecord.user_id == user_id)
    if loji:   query = query.filter(LogbookRecord.loji.contains(loji))
    if daerah: query = query.filter(LogbookRecord.daerah == daerah)
    records = query.order_by(LogbookRecord.tarikh, LogbookRecord.masa).all()
    db.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "No", "Tarikh", "Loji", "Daerah", "Masa", "Shift", "Masa Raw (OCR)",
        "Flow (m³/jam)", "pH Mentah", "NTU Mentah", "Warna Mentah", "Al (ppm)",
        "Fe Mentah (ppm)", "Mn Mentah (ppm)", "Cl- (ppm)", 
        "Tangki Flok pH",
        "pH Tangki Mendap", "NTU Tangki Mendap", "Warna Tangki Mendap",
        "pH Selepas Tapis", "NTU Selepas Tapis", "Warna Selepas Tapis", "Res.Al Tapis",
        "pH Bersih", "NTU Bersih", "Warna Bersih",
        "Fe Bersih (ppm)", "Res.Al Bersih (ppm)", "Mn Bersih (ppm)",
        "Res.F (ppm)", "Res.Cl2 (ppm)", "Cl Bersih (ppm)",
        "Kapur (pre) dos kg/jam", "Kapur (pre) ppm",
        "Kapur (post) dos kg/jam", "Kapur (post) ppm",
        "Koagulan dos lit/jam", "Koagulan ppm",
        "Polymer dos lit/jam", "Polymer ppm",
        "Klorin (Pre) dos kg/jam", "Klorin (Pre) ppm",
        "Klorin (Post) dos kg/jam", "Klorin (Post) ppm",
        "Fluoride dos lt/jam", "Fluoride ppm",
        "Paras Air (m)",
        "Status", "OCR Accuracy (%)", "Data Diedit",
    ])
    for i, r in enumerate(records, 1):
        acc_str    = f"{r.ocr_accuracy}%" if r.ocr_accuracy is not None else "-"
        status_str = (r.status or "") + (" ⚠LOW" if r.ocr_accuracy and r.ocr_accuracy < 80 else "")
        writer.writerow([
            i, r.tarikh, r.loji, r.daerah, r.masa, r.shift, r.masa_raw,
            r.flow, r.ph_mentah, r.ntu_mentah, r.warna_mentah, r.al,
            r.fe_mentah, r.mn_mentah, r.cl_mentah, 
            r.ph_flok,
            r.ph_tangki, r.ntu_tangki, r.warna_tangki,
            r.ph_tapis, r.ntu_tapis, r.warna_tapis, r.res_al_tapis,
            r.ph_bersih, r.ntu_bersih, r.warna_bersih,
            r.fe_bersih, r.res_al_bersih, r.mn_bersih,
            r.res_f_bersih, r.res_cl2, r.cl_bersih,
            r.kapur_pre, r.kapur_pre_ppm,
            r.kapur_post, r.kapur_post_ppm,
            r.koagulan_lit, r.koagulan_ppm,
            r.polymer_lit, r.polymer_ppm,
            r.klorin_pre, r.klorin_pre_ppm,
            r.klorin_post, r.klorin_post_ppm,
            r.fluoride, r.fluoride_ppm,
            r.paras_air,
            status_str, acc_str, "Ya" if r.is_edited else "Tidak",
        ])

    output.seek(0)
    fname = f"logbook_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fname}"}
    )


@app.get("/stats")
def get_stats(
    request: Request,
    user=Depends(get_current_user),
):
    db       = SessionLocal()
    base     = db.query(LogbookRecord).filter(LogbookRecord.user_id == user["id"])
    total    = base.count()
    normal   = base.filter(LogbookRecord.status == "Normal").count()
    amaran   = base.filter(LogbookRecord.status == "Amaran").count()
    kritikal = base.filter(LogbookRecord.status == "Kritikal").count()
    low_acc  = base.filter(
        LogbookRecord.ocr_accuracy != None,
        LogbookRecord.ocr_accuracy < 80
    ).count()
    db.close()
    return {
        "total_records": total, "normal": normal,
        "amaran": amaran, "kritikal": kritikal,
        "low_accuracy": low_acc,
    }


if __name__=="__main__":
    port=int(os.environ.get("PORT",8000))

    uvicorn.run(
        "main_v5:app",
        host="0.0.0.0",
        port=port
)
    
