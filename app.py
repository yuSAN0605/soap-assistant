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

logging.basicConfig(level=logging.DEBUG)
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

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "ok", "api_key_set": bool(GEMINI_API_KEY)}

@app.post("/api/generate-soap", response_model=SOAPResponse)
async def generate_soap(request: SOAPRequest):
    logger.info("=== generate_soap called ===")
    
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    
    input_text = request.inputText or ""
    karte_images = request.karteImages or []
    memo_images = request.memoImages or []
    
    logger.info(f"Input: text={bool(input_text)}, karte={len(karte_images)}, memo={len(memo_images)}")
    
    if not input_text and not karte_images and not memo_images:
        raise HTTPException(status_code=400, detail="At least one input required")
    
    raw_text = ""
    try:
        prompt = """あなたは理学療法士向けの専門カルテ（SOAP）記録生成AIです。提供された情報を分析し、以下のJSON形式で必ず返してください。

【最重要ルール】
- 「O」と「A」を分けず「oa」キーに統合
- 以下の5つのキーのみ返す：progress, notice, s, oa, p
- Markdown記号やコードブロックは含めない

【出力形式】
{"progress":"...","notice":"...","s":"...","oa":"...","p":"..."}
"""
        
        if input_text:
            prompt += f"\n\n【入力情報】\n{input_text}"
        
        parts = [prompt]
        
        logger.info(f"Processing {len(karte_images)} karte images...")
        for i, img_b64 in enumerate(karte_images):
            try:
                cleaned = clean_base64(img_b64)
                if cleaned:
                    image_bytes = base64.b64decode(cleaned)
                    parts.append({
                        "mime_type": "image/jpeg",
                        "data": image_bytes
                    })
                    logger.info(f"  karte image {i}: {len(image_bytes)} bytes")
            except Exception as e:
                logger.error(f"  karte image {i} failed: {e}")
        
        logger.info(f"Processing {len(memo_images)} memo images...")
        for i, img_b64 in enumerate(memo_images):
            try:
                cleaned = clean_base64(img_b64)
                if cleaned:
                    image_bytes = base64.b64decode(cleaned)
                    parts.append({
                        "mime_type": "image/jpeg",
                        "data": image_bytes
                    })
                    logger.info(f"  memo image {i}: {len(image_bytes)} bytes")
            except Exception as e:
                logger.error(f"  memo image {i} failed: {e}")
        
        logger.info(f"Calling Gemini API with {len(parts)} parts...")
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = await asyncio.to_thread(
            model.generate_content,
            parts,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1,
                "max_output_tokens": 4096
            }
        )
        
        raw_text = response.text.strip() if response and response.text else ""
        logger.info(f"API response length: {len(raw_text)}")
        
        if not raw_text:
            raise ValueError("Empty response from API")
        
        # JSON 抽出
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()
        
        result = json.loads(raw_text)
        logger.info("JSON parsed successfully")
        
        return SOAPResponse(
            progress=result.get("progress", ""),
            notice=result.get("notice", ""),
            s=result.get("s", ""),
            oa=result.get("oa", ""),
            p=result.get("p", "")
        )
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Raw text: {raw_text[:500]}")
        raise HTTPException(status_code=500, detail="JSON parse error")
    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))