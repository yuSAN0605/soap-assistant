import os
import json
import re
import asyncio
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai

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

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

class GenerateSoapRequest(BaseModel):
    inputText: str = ""
    karteImage: Optional[str] = None
    karteImages: Optional[List[str]] = Field(default_factory=list)
    memoImage: Optional[str] = None
    memoImages: Optional[List[str]] = Field(default_factory=list)
    attachedFiles: List[dict] = Field(default_factory=list)

class SoapResponse(BaseModel):
    progress: str = ""
    notice: str = ""
    s: str = ""
    oa: str = ""
    p: str = ""

CLEAN_MARKDOWN_REGEX = re.compile(r'^```(?:json)?\s*|\s*```$', re.MULTILINE)
EXTRACT_JSON_REGEX = re.compile(r'\{.*\}', re.DOTALL)

def clean_base64_data(data_str: str) -> str:
    if not data_str:
        return ""
    if "," in data_str:
        return data_str.split(",")[1]
    return data_str

def validate_and_fix_output(result: dict) -> tuple:
    raw_progress = result.get("progress", "")
    cleaned_progress = re.sub(r'^(?:[＊\*]?経過\s*[\r\n]*)+', '', raw_progress).strip()
    
    notice = result.get("notice", "").strip()
    s_content = result.get("s", "").strip()
    oa_content = result.get("oa", "").strip()
    
    if not oa_content or "ROM-T" not in oa_content:
        o_part = result.get("o", "").strip()
        a_part = result.get("a", "").strip()
        
        base_o = o_part if o_part else "ROM-T:\nMMT:\nPain:\nalignment:\ngait:\nその他："
        if a_part:
            if "その他：" in base_o:
                oa_content = base_o.replace("その他：", f"その他：{a_part}")
            else:
                oa_content = f"{base_o}\nその他：{a_part}"
        else:
            oa_content = base_o
    
    if "ROM-T" not in oa_content:
        oa_content = f"ROM-T:\nMMT:\nPain:\nalignment:\ngait:\nその他：\n{oa_content}"

    fixed_p = "#1 関節可動域訓練 #2 筋力強化訓練 #3 バランス訓練 #4 自主トレーニング指導"

    return cleaned_progress, notice, s_content, oa_content, fixed_p

@app.post("/api/generate-soap", response_model=SoapResponse)
async def generate_soap(request: GenerateSoapRequest):
    try:
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="Gemini API Key is not configured.")
        
        input_text = request.inputText or ""
        
        karte_images = request.karteImages or []
        if not karte_images and request.karteImage:
            karte_images = [request.karteImage]

        memo_images = request.memoImages or []
        if not memo_images and request.memoImage:
            memo_images = [request.memoImage]

        attached_files = request.attachedFiles or []
        
        if not input_text and not karte_images and not memo_images and not attached_files:
            raise HTTPException(status_code=400, detail="At least one input must be provided.")

        promptText = f"""あなたは理学療法士向けの専門カルテ（SOAP）記録生成AIです。提供された画像や情報を分析し、理学療法記録として正確に構造化したJSONデータを作成してください。

■【最重要ルール】出力フォーマットについて：
1. 「O」や「A」という独立した項目・見出しは絶対に作成しないでください。
2. 客観的所見と臨床推論は必ず「oa」というキー名の中で統合し、以下のフォーマットを厳守してください：
ROM-T:
MMT:
Pain:
alignment:
gait:
その他：
（※「その他：」の後に、従来のAに相当する臨床推論や考察を記載してください）

■ 各項目の記載ルール：
【progress（経過）】
- 出力の先頭は必ず以下のテキストから始めてください：
算定区分：運動器リハビリテーション料(Ⅰ)
実施区分：2単位
実施時間：
実施者：長岡
本日より理学療法開始
【現病歴】[現病歴の内容]
- 画像所見が存在する場合は必ず直前で改行し、以下のように別行で記載：
  【画像所見】
  X線：[所見内容]（撮影日）
  MRI：[所見内容]（撮影日）
- ない場合は【画像所見】の行を出力しない。

【notice（注意点）】
- 入力情報内に存在する項目のみ「既往歴：」「体重：」「仕事：」の形式で記載。ない場合は空文字。

【s（Subjective）】
- 患者自身の言葉のみ。鍵カッコ「 」を使用。
- 測定結果や所見の中で「特記なし」や該当するものがない項目がある場合、「特記なし」や「無記載」などの文字は一切出力せず、項目名の後ろを空欄のままにしてください。

【p（Plan）】
- 理由は一切記載せず、以下の1行のみで出力してください：
#1 関節可動域訓練 #2 筋力強化訓練 #3 バランス訓練 #4 自主トレーニング指導

【重要】以下のJSON形式で**必ず**レスポンスしてください。他の説明やマークダウンコードブロック（```json など）は一切含めず、純粋なJSON文字列のみを出力してください。

{{
  "progress": "...",
  "notice": "...",
  "s": "...",
  "oa": "ROM-T:...\nMMT:...\nPain:...\nalignment:...\ngait:...\nその他：...",
  "p": "#1 関節可動域訓練 #2 筋力強化訓練 #3 バランス訓練 #4 自主トレーニング指導"
}}"""

        partsArr = [{"text": promptText}]
        
        if input_text:
            partsArr.append({"text": f"■ 入力テキストメモ:\n{input_text}"})
        
        for img_data in karte_images:
            cleaned_b64 = clean_base64_data(img_data)
            if cleaned_b64:
                partsArr.append({
                    "inline_data": {"mime_type": "image/jpeg", "data": cleaned_b64}
                })
        
        for img_data in memo_images:
            cleaned_b64 = clean_base64_data(img_data)
            if cleaned_b64:
                partsArr.append({
                    "inline_data": {"mime_type": "image/jpeg", "data": cleaned_b64}
                })
        
        for fileObj in attached_files:
            if isinstance(fileObj, dict) and fileObj.get("data"):
                cleaned_b64 = clean_base64_data(fileObj.get("data"))
                if cleaned_b64:
                    partsArr.append({
                        "inline_data": {
                            "mime_type": fileObj.get("mimeType", "application/octet-stream"),
                            "data": cleaned_b64
                        }
                    })
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        def _call_gemini():
            return model.generate_content(
                partsArr,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=4096
                )
            )

        response = await asyncio.to_thread(_call_gemini)
        
        raw_text = response.text.strip() if response.text else ""
        if not raw_text:
            raise ValueError("Empty response from Gemini API")
        
        raw_text = CLEAN_MARKDOWN_REGEX.sub('', raw_text).strip()
        
        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError:
            match = EXTRACT_JSON_REGEX.search(raw_text)
            if match:
                try:
                    result = json.loads(match.group(0))
                except json.JSONDecodeError:
                    result = {"progress": "", "notice": "", "s": "", "oa": "", "p": ""}
            else:
                result = {"progress": "", "notice": "", "s": "", "oa": "", "p": ""}
        
        cleaned_progress, notice_val, s_val, oa_val, fixed_p = validate_and_fix_output(result)

        return SoapResponse(
            progress=cleaned_progress,
            notice=notice_val,
            s=s_val,
            oa=oa_val,
            p=fixed_p
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "static_dir_exists": STATIC_DIR.exists(),
        "index_html_exists": (STATIC_DIR / "index.html").exists()
    }

@app.get("/")
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail=f"index.html not found at {index_file}")
    return FileResponse(str(index_file), media_type="text/html")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")