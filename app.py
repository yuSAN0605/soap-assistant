import os
import json
import asyncio
import logging
import base64
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app.mount("/static", StaticFiles(directory="static"), name="static")

class SOAPRequest(BaseModel):
    inputText: str = ""
    karteImages: Optional[List[str]] = Field(default_factory=list)
    memoImages: Optional[List[str]] = Field(default_factory=list)

class SOAPResponse(BaseModel):
    progress: str = ""
    notice: str = ""
    s: str = ""
    oa: str = ""
    p: str = ""

def clean_base64(data_str: str) -> str:
    if not data_str:
        return ""
    if "," in data_str:
        return data_str.split(",")[1]
    return data_str

def generate_prompt(num_karte: int, num_memo: int, input_text: str) -> str:
    prompt = """あなたは理学療法士向けの専門カルテ（SOAP）記録生成AIです。
提供された画像とテキストを分析し、以下のフォーマットとルールを厳守してJSON形式で必ず返してください。

【出力フォーマット・記載ルール】

1. progress（経過）：
- 出力の先頭は必ず以下のテキストから始めてください：
算定区分：運動器リハビリテーション料(Ⅰ)
実施区分：2単位
実施時間：
実施者：長岡
本日より理学療法開始
【現病歴】[現病歴の内容]
- 画像所見が存在する場合は必ず直前で改行し、以下のように別行で記載してください：
  【画像所見】
  X線：[所見内容]（撮影日）
  MRI：[所見内容]（撮影日）
- ない場合は【画像所見】の行を出力しないでください。

2. notice（注意点）：
- 入力情報内に存在する項目のみ「既往歴：」「体重：」「仕事：」の形式で記載してください。ない場合は出力しません。

3. s：
- 患者自身の言葉のみを抽出してください。必ず鍵カッコ「 」を使用してください。

4. oa（Objective / Assessment統合）：
以下の項目名と形式で出力してください。測定結果や所見の中で「特記なし」や該当するものがない項目がある場合、「特記なし」や「無記載」などの文字は一切出力せず、項目名の後ろを空欄（値なし）のままにしてください。
ROM-T:
MMT:
Pain・specialtest:
alignment:
gait:
その他：
（※「その他：」の後に、臨床推論や考察を記載してください）

5. p（Plan）：
- 理由は一切記載せず、以下の1行のみ（改行せず、スペース区切りの横一列）で出力してください：
#1 関節可動域訓練 #2 筋力強化訓練 #3 バランス訓練 #4 自主トレーニング指導

【最重要ルール】
- 「O」と「A」を分けず「oa」キーに統合すること
- 複数画像から得られた情報は統合し、矛盾する場合は最新情報を優先すること
- 以下の5つのキーのみ返すこと：progress, notice, s, oa, p
- Markdown記号やコードブロックは含めないこと

【出力形式（必ずこのJSON構造にすること）】
{"progress":"...","notice":"...","s":"...","oa":"...","p":"..."}
"""
    
    if num_karte > 0:
        prompt += f"\n- カルテ画像（{num_karte}枚）：院内記録、検査結果、画像診断などとして解析"
    if num_memo > 0:
        prompt += f"\n- メモ画像（{num_memo}枚）：手書きメモ、臨床情報などとして解析"
    if input_text:
        prompt += f"\n\n【追加入力情報】\n{input_text}"
    
    return prompt

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "api_key_set": bool(GEMINI_API_KEY),
        "sdk": "google-genai",
        "model": "gemini-3.6-flash",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/generate-soap", response_model=SOAPResponse)
async def generate_soap(request: SOAPRequest):
    request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")[:17]
    logger.info(f"[{request_id}] === generate_soap called ===")
    
    if not client:
        logger.error(f"[{request_id}] API key not configured")
        raise HTTPException(status_code=500, detail="API key not configured")
    
    input_text = request.inputText or ""
    karte_images = request.karteImages or []
    memo_images = request.memoImages or []
    
    logger.info(f"[{request_id}] Input: text={bool(input_text)}, karte={len(karte_images)}, memo={len(memo_images)}")
    
    if not input_text and not karte_images and not memo_images:
        raise HTTPException(status_code=400, detail="At least one input required")
    
    total_size = sum(len(img) for img in karte_images + memo_images)
    if total_size > 50 * 1024 * 1024:
        logger.warning(f"[{request_id}] Total image size exceeds 50MB")
        raise HTTPException(status_code=413, detail="Images too large (max 50MB total)")
    
    raw_text = ""
    try:
        prompt = generate_prompt(len(karte_images), len(memo_images), input_text)
        
        contents = [types.Part.from_text(text=prompt)]
        
        logger.info(f"[{request_id}] Processing {len(karte_images)} karte images...")
        for i, img_b64 in enumerate(karte_images):
            try:
                cleaned = clean_base64(img_b64)
                if cleaned:
                    image_bytes = base64.b64decode(cleaned)
                    contents.append(
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/jpeg"
                        )
                    )
            except Exception as e:
                logger.error(f"[{request_id}] karte image {i} decode failed: {e}")
                raise HTTPException(status_code=400, detail="Image decode error")
        
        logger.info(f"[{request_id}] Processing {len(memo_images)} memo images...")
        for i, img_b64 in enumerate(memo_images):
            try:
                cleaned = clean_base64(img_b64)
                if cleaned:
                    image_bytes = base64.b64decode(cleaned)
                    contents.append(
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/jpeg"
                        )
                    )
            except Exception as e:
                logger.error(f"[{request_id}] memo image {i} decode failed: {e}")
                raise HTTPException(status_code=400, detail="Image decode error")
        
        logger.info(f"[{request_id}] Calling Gemini API (gemini-3.6-flash)...")
        
        max_retries = 2
        response = None
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.models.generate_content,
                        model='gemini-3.6-flash',
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1,
                            max_output_tokens=8192
                        )
                    ),
                    timeout=90.0
                )
                logger.info(f"[{request_id}] API response received (attempt {attempt + 1})")
                break
            except asyncio.TimeoutError:
                logger.warning(f"[{request_id}] API timeout (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    raise HTTPException(status_code=504, detail="API timeout")
                await asyncio.sleep(2)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    raise HTTPException(status_code=429, detail="API rate limit exceeded")
                if "404" in error_str or "not found" in error_str.lower():
                    raise HTTPException(status_code=400, detail="Model not available")
                raise
        
        if not response:
            raise ValueError("No response from API")
        
        raw_text = response.text.strip() if response.text else ""
        
        if not raw_text:
            raise ValueError("Empty response from API")
        
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()
        
        result = json.loads(raw_text)
        
        required_keys = {"progress", "notice", "s", "oa", "p"}
        missing_keys = required_keys - set(result.keys())
        if missing_keys:
            for key in missing_keys:
                result[key] = ""
        
        return SOAPResponse(
            progress=result.get("progress", ""),
            notice=result.get("notice", ""),
            s=result.get("s", ""),
            oa=result.get("oa", ""),
            p=result.get("p", "")
        )
    
    except json.JSONDecodeError as e:
        logger.error(f"[{request_id}] JSON parse error: {e}")
        raise HTTPException(status_code=500, detail="JSON parse error")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error")