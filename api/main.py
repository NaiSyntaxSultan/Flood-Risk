import uvicorn
from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel
import logging
import pandas as pd

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
MAPPING_PATH    = MODELS_DIR / "mapping.pkl"
COLUMNS_PATH    = MODELS_DIR / "columns.pkl"

app = FastAPI(
    title="Flood Risk Prediction API",
    description="API for predicting flood risk (Low/Medium/High) from MaxRain and Season",
    version="1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # (แนะนำจำกัดโดเมนจริงก่อน deploy)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Flood Risk Prediction API!"}


# =========================
# 1) Prediction ผ่าน query
# =========================
@app.post('/prediction', tags=["predictions"])
async def get_prediction(MaxRain: float, Season_Cool: int, Season_Hot: int, Season_Rainy: int):

    # Load model/meta
    model = load(BEST_MODEL_PATH)
    mapping = load(MAPPING_PATH)
    columns = list(load(COLUMNS_PATH))

    feats = {
        "MaxRain": float(MaxRain),
        "Season_Cool": int(Season_Cool),
        "Season_Hot": int(Season_Hot),
        "Season_Rainy": int(Season_Rainy),
    }

    # ใช้ DataFrame พร้อมชื่อคอลัมน์
    X = pd.DataFrame([[feats[c] for c in columns]], columns=columns)
    pred_id = int(model.predict(X)[0])

    label_map = mapping.get("Outcome Variable", mapping) if isinstance(mapping, dict) else {}
    try:
        label_map = {int(k): v for k, v in label_map.items()}
    except Exception:
        pass
    prediction = label_map.get(pred_id, str(pred_id))

    return {"prediction": prediction, "label_id": pred_id, "columns_order": columns}


# =========================
# 2) Prediction ผ่าน JSON body (Season one-hot)
# =========================
class PredictionInput(BaseModel):
    MaxRain: float
    Season_Cool: int
    Season_Hot: int
    Season_Rainy: int

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.post('/prediction_web', tags=["predictions"])
async def get_prediction(request: Request, input_data: PredictionInput = Body(...)):
    try:
        model = load(BEST_MODEL_PATH)
        mapping = load(MAPPING_PATH)
        columns = list(load(COLUMNS_PATH))

        payload = input_data.dict()
        feats = {
            "MaxRain": float(payload["MaxRain"]),
            "Season_Cool": int(payload["Season_Cool"]),
            "Season_Hot": int(payload["Season_Hot"]),
            "Season_Rainy": int(payload["Season_Rainy"]),
        }

        X = pd.DataFrame([[feats[c] for c in columns]], columns=columns)
        pred_id = int(model.predict(X)[0])

        label_map = mapping.get("Outcome Variable", mapping) if isinstance(mapping, dict) else {}
        try:
            label_map = {int(k): v for k, v in label_map.items()}
        except Exception:
            pass
        prediction = label_map.get(pred_id, str(pred_id))

        return {"prediction": prediction, "label_id": pred_id, "columns_order": columns, "status": "success"}

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# =====================================================================
# 3) Prediction ผ่าน month, province, MaxRain
# =====================================================================
# Province sets
north = {"เชียงใหม่","เชียงราย","ลำพูน","ลำปาง","แพร่","น่าน","พะเยา","แม่ฮ่องสอน",
         "ตาก","อุตรดิตถ์","พิษณุโลก","สุโขทัย","กำแพงเพชร","พิจิตร","เพชรบูรณ์",
         "นครสวรรค์","อุทัยธานี"}
northeast = {"ขอนแก่น","อุดรธานี","อุบลราชธานี","นครราชสีมา","บุรีรัมย์","สุรินทร์",
             "ศรีสะเกษ","ร้อยเอ็ด","ยโสธร","มหาสารคาม","กาฬสินธุ์","สกลนคร","นครพนม",
             "หนองคาย","เลย","หนองบัวลำภู","บึงกาฬ","มุกดาหาร","ชัยภูมิ","อำนาจเจริญ"}
central = {"กรุงเทพมหานคร","นนทบุรี","ปทุมธานี","พระนครศรีอยุธยา","สระบุรี","ลพบุรี",
           "อ่างทอง","สิงห์บุรี","ชัยนาท","สุพรรณบุรี","นครปฐม","สมุทรสาคร",
           "สมุทรปราการ","สมุทรสงคราม","นครนายก"}
east = {"ชลบุรี","ระยอง","จันทบุรี","ตราด","ฉะเชิงเทรา","ปราจีนบุรี","สระแก้ว"}
west = {"กาญจนบุรี","ราชบุรี","เพชรบุรี","ประจวบคีรีขันธ์"}
south_andaman = {"ระนอง","พังงา","ภูเก็ต","กระบี่","ตรัง","สตูล"}
south_gulf = {"ชุมพร","สุราษฎร์ธานี","นครศรีธรรมราช","พัทลุง","สงขลา","ปัตตานี","ยะลา","นราธิวาส"}

REGION_SETS = {
    "North": north,
    "Northeast": northeast,
    "Central": central,
    "East": east,
    "West": west,
    "South_Andaman": south_andaman,
    "South_Gulf": south_gulf,
}

def _norm_th(s: str) -> str:
    return (s or "").strip().lower()

def province_to_region(province_th: str) -> str:
    p = _norm_th(province_th)
    for region, provs in REGION_SETS.items():
        if any(_norm_th(x) == p for x in provs):
            return region
    if "กรุงเทพ" in p:
        return "Central"
    raise HTTPException(status_code=400, detail=f"ไม่พบภาคสำหรับจังหวัด: {province_th}")

def month_region_to_season(month: int, region: str) -> str:
    # Mainland (มี Cool)
    if region in {"North","Northeast","Central","East","West"}:
        if month in {3,4,5}:
            return "Hot"
        elif month in {11,12,1,2}:
            return "Cool"
        else:
            return "Rainy"

    # South Andaman (ไม่มี Cool)
    if region == "South_Andaman":
        if month in {3,4}:
            return "Hot"
        else:
            return "Rainy"

    # South Gulf (ไม่มี Cool)
    if region == "South_Gulf":
        if month in {4,5}:
            return "Hot"
        else:
            return "Rainy"

    return "Rainy"

def season_to_onehot(season: str) -> dict:
    s = (season or "").lower()
    return {
        "Season_Cool":   1 if s == "cool"   else 0,
        "Season_Hot":    1 if s == "hot"    else 0,
        "Season_Rainy":  1 if s == "rainy"  else 0,
    }

class SimpleInput(BaseModel):
    month: int
    province: str
    MaxRain: float

@app.post("/predict_from_fields", tags=["predictions"])
async def predict_from_fields(input_data: SimpleInput):
    try:
        region = province_to_region(input_data.province)
        season = month_region_to_season(input_data.month, region)
        onehot = season_to_onehot(season)

        model = load(BEST_MODEL_PATH)
        mapping = load(MAPPING_PATH)
        columns = list(load(COLUMNS_PATH))

        feats = {
            "MaxRain": float(input_data.MaxRain),
            "Season_Cool": int(onehot["Season_Cool"]),
            "Season_Hot": int(onehot["Season_Hot"]),
            "Season_Rainy": int(onehot["Season_Rainy"]),
        }

        X = pd.DataFrame([[feats[c] for c in columns]], columns=columns)
        pred_id = int(model.predict(X)[0])

        label_map = mapping.get("Outcome Variable", mapping) if isinstance(mapping, dict) else {}
        try:
            label_map = {int(k): v for k, v in label_map.items()}
        except Exception:
            pass
        prediction = label_map.get(pred_id, str(pred_id))

        return {
            "inputs": {
                "month": input_data.month,
                "province": input_data.province,
                "region": region,
                "season": season,
                **onehot,
                "MaxRain": input_data.MaxRain,
            },
            "columns_order": columns,
            "label_id": pred_id,
            "prediction": prediction,
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
