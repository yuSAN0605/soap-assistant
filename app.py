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
import google.generativeai as genai
from datetime import datetime

# ロギング設定
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
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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

def estimate_api_cost(num_images: int) -> str:
    base_tokens = 1000
    image_tokens = num_images * 1000
    total = base_tokens + image_tokens
    return f"~{total}tokens"

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "api_key_set": bool(GEMINI_API_KEY),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/generate-soap", response_model=SOAPResponse)
async def generate_soap(request: SOAPRequest):
    request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")[:17]
    logger.info(f"[{request_id}] === generate_soap called ===")
    
    if not GEMINI_API_KEY:
        logger.error(f"[{request_id}] API key not configured")
        raise HTTPException(status_code=500, detail="API key not configured")
    
    input_text = request.inputText or ""
    karte_images = request.karteImages or []
    memo_images = request.memoImages or []
    
    logger.info(f"[{request_id}] Input: text={bool(input_text)}, karte={len(karte_images)}, memo={len(memo_images)}, cost={estimate_api_cost(len(karte_images) + len(memo_images))}")
    
    if not input_text and not karte_images and not memo_images:
        raise HTTPException(status_code=400, detail="At least one input required")
    
    total_size = sum(len(img) for img in karte_images + memo_images)
    if total_size > 50 * 1024 * 1024:
        logger.warning(f"[{request_id}] Total image size exceeds 50MB")
        raise HTTPException(status_code=413, detail="Images too large (max 50MB total)")
    
    raw_text = ""
    try:
        prompt = generate_prompt(len(karte_images), len(memo_images), input_text)
        parts = [prompt]
        
        logger.info(f"[{request_id}] Processing {len(karte_images)} karte images...")
        for i, img_b64 in enumerate(karte_images):
            try:
                cleaned = clean_base64(img_b64)
                if cleaned:
                    image_bytes = base64.b64decode(cleaned)
                    parts.append({
                        "mime_type": "image/jpeg",
                        "data": image_bytes
                    })
                    logger.debug(f"[{request_id}] karte image {i}: {len(image_bytes) / 1024:.1f}KB")
            except Exception as e:
                logger.error(f"[{request_id}] karte image {i} failed: {e}")
                raise HTTPException(status_code=400, detail=f"Image decode failed: {str(e)}")
        
        logger.info(f"[{request_id}] Processing {len(memo_images)} memo images...")
        for i, img_b64 in enumerate(memo_images):
            try:
                cleaned = clean_base64(img_b64)
                if cleaned:
                    image_bytes = base64.b64decode(cleaned)
                    parts.append({
                        "mime_type": "image/jpeg",
                        "data": image_bytes
                    })
                    logger.debug(f"[{request_id}] memo image {i}: {len(image_bytes) / 1024:.1f}KB")
            except Exception as e:
                logger.error(f"[{request_id}] memo image {i} failed: {e}")
                raise HTTPException(status_code=400, detail=f"Image decode failed: {str(e)}")
        
        logger.info(f"[{request_id}] Calling Gemini API with gemini-1.5-flash...")
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        model.generate_content,
                        parts,
                        generation_config={
                            "response_mime_type": "application/json",
                            "temperature": 0.1,
                            "max_output_tokens": 8192
                        }
                    ),
                    timeout=90.0
                )
                logger.info(f"[{request_id}] API response received (attempt {attempt + 1})")
                break
            except asyncio.TimeoutError:
                logger.warning(f"[{request_id}] API timeout (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    raise HTTPException(status_code=504, detail="API timeout - please try again")
                await asyncio.sleep(2)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logger.error(f"[{request_id}] Rate limit exceeded: {e}")
                    raise HTTPException(status_code=429, detail="API rate limit exceeded")
                raise
        
        raw_text = response.text.strip() if response and response.text else ""
        logger.info(f"[{request_id}] API response length: {len(raw_text)}")
        
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
            logger.warning(f"[{request_id}] Missing keys: {missing_keys}")
            for key in missing_keys:
                result[key] = ""
        
        logger.info(f"[{request_id}] JSON parsed successfully")
        
        return SOAPResponse(
            progress=result.get("progress", ""),
            notice=result.get("notice", ""),
            s=result.get("s", ""),
            oa=result.get("oa", ""),
            p=result.get("p", "")
        )
    
    except json.JSONDecodeError as e:
        logger.error(f"[{request_id}] JSON parse error: {e}")
        logger.error(f"[{request_id}] Raw text preview: {raw_text[:300]}")
        raise HTTPException(status_code=500, detail="API response parse failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {type(e).__name__}")

def generate_prompt(num_karte: int, num_memo: int, input_text: str) -> str:
    prompt = """あなたは理学療法士向けの専門カルテ（SOAP）記録生成AIです。
提供された画像とテキストを段階的に分析し、以下のJSON形式で必ず返してください。

【画像解析の指示】"""
    
    if num_karte > 0:
        prompt += f"""
- カルテ画像（{num_karte}枚）：院内記録、診療録、検査結果として解析
  各画像から：患者情報、既往歴、体重、検査所見、画像診断を抽出"""
    
    if num_memo > 0:
        prompt += f"""
- メモ画像（{num_memo}枚）：手書きメモ、申し送り、臨床情報として解析
  各画像から：主訴、症状、動作制限、特記事項を抽出"""
    
    prompt += """

【最重要ルール】
- 「O」と`A`を分けず「oa」キーに統合
- 複数画像から得られた情報は統合し、矛盾する場合は最新情報を優先
- 以下の5つのキーのみ返す：progress, notice, s, oa, p
- Markdown記号やコードブロックは含めない

【出力形式（必ずこの形）】
{"progress":"...","notice":"...","s":"...","oa":"...","p":"..."}
"""
    
    if input_text:
        prompt += f"\n【追加入力情報】\n{input_text}"
    
    return prompt