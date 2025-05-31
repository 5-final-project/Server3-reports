import os
import time
import logging
import re
import boto3
import markdown2
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from weasyprint import HTML

import aiohttp
import asyncio

# ----- 로그 설정 -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("meeting-summary")

# ----- 환경변수 로드 및 체크 -----
load_dotenv()
REQUIRED_ENV_VARS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "BUCKET_NAME"]
for v in REQUIRED_ENV_VARS:
    if not os.getenv(v):
        logger.error(f"환경변수 {v}가 누락되어 있습니다.")
        raise RuntimeError(f"환경변수 {v}가 누락되어 있습니다.")

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
REGION = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")
PDF_UPLOAD_URL = "https://team5opensearch.ap.loclx.io/documents/upload-without-s3"
QWEN_API_URL = "https://qwen3.ap.loclx.io/api/generate"

# ----- 데이터 모델 -----
class RelatedDoc(BaseModel):
    page_content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None

class ChunkItem(BaseModel):
    chunk_en: str
    related_docs: List[RelatedDoc]

class MeetingMeta(BaseModel):
    title: Optional[str] = None
    datetime: Optional[str] = None
    author: Optional[str] = None
    participants: Optional[List[str]] = None

class MeetingInput(BaseModel):
    text_stt: str
    meeting_meta: Optional[MeetingMeta] = None
    chunks: Optional[List[ChunkItem]] = None
    elapsed_time: Optional[float] = None
    error: Optional[str] = None

# ----- 줄바꿈 후처리 함수 -----
def add_newlines_between_sentences(text):
    # 1. 번호로 시작하는 새로운 항목 앞에 줄바꿈
    text = re.sub(r'([.?!])(\d+\.)', r'\1\n\2', text)
    # 2. 문장부호 후 띄어쓰기 없는 다음 문장에 줄바꿈
    text = re.sub(r'([.?!])([^\s\d])', r'\1\n\2', text)
    return text

# ----- LLM 처리 함수 -----
def clean_llm_output(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    # <think> 태그 제거
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # 문장 줄바꿈 추가
    text = add_newlines_between_sentences(text)
    return text

async def post_json(session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
    try:
        async with session.post(url, json=payload) as response:
            resp_text = await response.text()
            content_type = response.headers.get('Content-Type', '')
            try:
                if "application/json" in content_type:
                    return json.loads(resp_text)
                else:
                    return json.loads(resp_text)
            except Exception as e:
                logger.error(f"JSON 파싱 에러: {e} | 응답 본문: {resp_text[:300]}")
                raise Exception(f"JSON 파싱 실패. 본문: {resp_text[:300]}")
    except Exception as e:
        logger.error(f"API 요청 실패({url}): {e}")
        raise

async def async_llm_map_summary(chunk_text: str, session: aiohttp.ClientSession) -> str:
    payload = {
        "prompt": [{"role": "user", "content": f"아래 회의내용을 핵심을 중심으로 요약해주세요. \n핵심내용에 번호를 순서대로 붙여주세요. \n마지막에 추가 요약은 절대 하지 마세요.\n\n회의내용:\n{chunk_text}"}],
        "max_tokens": 30000,
        "temperature": 0.7,
        "top_p": 0.9
    }
    try:
        result = await post_json(session, QWEN_API_URL, payload)
        return clean_llm_output(result.get("response", ""))
    except Exception as e:
        logger.error(f"async_llm_map_summary 에러: {e}")
        raise

async def async_llm_combine_summary(map_summaries: list, session: aiohttp.ClientSession) -> str:
    combined_text = "\n".join(map_summaries)
    payload = {
        "prompt": [{"role": "user", "content": f"여러 회의 요약문이 아래에 나열되어 있습니다. 중복 없이 통합 정리해주세요. \n핵심내용에 번호를 순서대로 붙여주세요. \n마지막에 추가 요약은 절대 하지 마세요.\n\n{combined_text}"}],
        "max_tokens": 30000,
        "temperature": 0.7,
        "top_p": 0.9
    }
    try:
        result = await post_json(session, QWEN_API_URL, payload)
        return clean_llm_output(result.get("response", ""))
    except Exception as e:
        logger.error(f"async_llm_combine_summary 에러: {e}")
        raise

# [추가] Action Item map-reduce 방식 함수들
async def async_llm_map_action_items(summary_text: str, session: aiohttp.ClientSession) -> str:
    payload = {
        "prompt": [{"role": "user", "content": f"아래 요약문을 바탕으로 앞으로 해야할 일(To-Do List)을 작성해주세요. 각 list 항목에 번호를 붙여주세요. \n 마지막에 추가 요약은 절대 하지 마세요.\n\n{summary_text}"}],
        "max_tokens": 30000,
        "temperature": 0.7,
        "top_p": 0.9
    }
    try:
        result = await post_json(session, QWEN_API_URL, payload)
        return clean_llm_output(result.get("response", ""))
    except Exception as e:
        logger.error(f"async_llm_map_action_items 에러: {e}")
        raise

async def async_llm_combine_action_items(action_items_list: list, session: aiohttp.ClientSession) -> str:
    combined_text = "\n".join(action_items_list)
    payload = {
        "prompt": [{"role": "user", "content": f"앞으로 해야할 일들의 리스트가 아래에 나열되어 있습니다. 중복을 제거하고, 통합 정리해서 To-Do List만 번호와 함께 다시 출력해주세요. 마지막에 추가 요약은 하지 마세요.\n\n{combined_text}"}],
        "max_tokens": 30000,
        "temperature": 0.7,
        "top_p": 0.9
    }
    try:
        result = await post_json(session, QWEN_API_URL, payload)
        return clean_llm_output(result.get("response", ""))
    except Exception as e:
        logger.error(f"async_llm_combine_action_items 에러: {e}")
        raise

# ----- 보고서 생성 함수 -----
def render_markdown_report(summary, actions):
    return f"""# 회의 요약 보고서

## 요약
{summary}

---

## Action Items
{actions}
"""

def markdown_to_html(md):
    css = """
    <style>
        body { font-family: 'NanumGothic', 'Noto Sans KR', sans-serif; white-space: pre-line; }
        h1, h2, h3 { color: #1a237e; }
    </style>
    """
    html = markdown2.markdown(md, extras=["fenced-code-blocks", "tables"])
    return f"<html><head><meta charset='utf-8'>{css}</head><body>{html}</body></html>"

def html_to_pdf(html, pdf_path):
    HTML(string=html).write_pdf(pdf_path)

def upload_to_s3(file_path, bucket, key):
    try:
        s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=REGION)
        s3.upload_file(file_path, bucket, key)
        url = s3.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=3600)
        return url
    except Exception as e:
        logger.error(f"S3 업로드 에러: {e}")
        raise

# ----- FastAPI 앱 -----
app = FastAPI(title="회의 요약 PDF API", version="2.7")

@app.post("/report-json")
async def report_json(request: MeetingInput):
    try:
        t0 = time.perf_counter()
        logger.info("요약 요청 수신")

        # Step 1. 텍스트 분할
        t_split_start = time.perf_counter()
        splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=100)
        chunks = splitter.split_text(request.text_stt)
        t_split_end = time.perf_counter()
        split_time = t_split_end - t_split_start
        logger.info(f"[STEP 1] 텍스트 분할 완료 (chunk={len(chunks)}, 소요={split_time:.2f}s, 누적={t_split_end-t0:.2f}s)")
        if not chunks:
            raise HTTPException(status_code=400, detail="요약할 텍스트가 없습니다.")

        async with aiohttp.ClientSession() as session:
            # Step 2. Map 요약
            t_map_start = time.perf_counter()
            try:
                map_summaries = await asyncio.gather(*[async_llm_map_summary(c, session) for c in chunks])
            except Exception as e:
                logger.error(f"[STEP 2] map 요약 실패: {e}")
                raise HTTPException(status_code=500, detail=f"map 요약 실패: {e}")
            t_map_end = time.perf_counter()
            map_time = t_map_end - t_map_start
            logger.info(f"[STEP 2] map 요약 완료 (소요={map_time:.2f}s, 누적={t_map_end-t0:.2f}s)")

            # Step 3. Combine 요약
            t_combine_start = time.perf_counter()
            try:
                if len(map_summaries) == 1:
                    full_summary = map_summaries[0]
                else:
                    full_summary = await async_llm_combine_summary(map_summaries, session)
            except Exception as e:
                logger.error(f"[STEP 3] combine 요약 실패: {e}")
                raise HTTPException(status_code=500, detail=f"combine 요약 실패: {e}")
            t_combine_end = time.perf_counter()
            combine_time = t_combine_end - t_combine_start
            logger.info(f"[STEP 3] combine 요약 완료 (소요={combine_time:.2f}s, 누적={t_combine_end-t0:.2f}s)")

            # Step 4. Action Items - Map-Reduce 방식
            t_action_start = time.perf_counter()
            try:
                # 4-1. 각 chunk 요약별로 action item 추출
                map_action_items = await asyncio.gather(
                    *[async_llm_map_action_items(summary, session) for summary in map_summaries]
                )
                # 4-2. 여러 action item을 LLM으로 통합
                if len(map_action_items) == 1:
                    action_items = map_action_items[0]
                else:
                    action_items = await async_llm_combine_action_items(map_action_items, session)
            except Exception as e:
                logger.error(f"[STEP 4] action item(map-reduce) 생성 실패: {e}")
                raise HTTPException(status_code=500, detail=f"action item(map-reduce) 생성 실패: {e}")
            t_action_end = time.perf_counter()
            action_time = t_action_end - t_action_start
            logger.info(f"[STEP 4] action items(map-reduce) 생성 완료 (소요={action_time:.2f}s, 누적={t_action_end-t0:.2f}s)")

            # Step 5. PDF 생성
            ts = int(time.time())
            pdf_path = os.path.abspath(f"meeting_report_{ts}.pdf")
            md = render_markdown_report(full_summary, action_items)
            html = markdown_to_html(md)
            t_pdf_start = time.perf_counter()
            try:
                html_to_pdf(html, pdf_path)
            except Exception as e:
                logger.error(f"[STEP 5] PDF 생성 실패: {e}")
                raise HTTPException(status_code=500, detail=f"PDF 생성 실패: {e}")
            t_pdf_end = time.perf_counter()
            pdf_time = t_pdf_end - t_pdf_start
            logger.info(f"[STEP 5] PDF 생성 완료 (소요={pdf_time:.2f}s, 누적={t_pdf_end-t0:.2f}s)")

            # Step 6. S3 업로드
            t_s3_start = time.perf_counter()
            try:
                s3_key = f"reports/meeting_report_{ts}.pdf"
                download_url = upload_to_s3(pdf_path, BUCKET_NAME, s3_key)
            except Exception as e:
                logger.error(f"[STEP 6] S3 업로드 실패: {e}")
                raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {e}")
            t_s3_end = time.perf_counter()
            s3_time = t_s3_end - t_s3_start
            logger.info(f"[STEP 6] S3 업로드 완료 (소요={s3_time:.2f}s, 누적={t_s3_end-t0:.2f}s)")

            # Step 7. PDF 문서 등록 API 호출
            pdf_doc_id = ""
            t_api_start = time.perf_counter()
            try:
                with open(pdf_path, "rb") as f:
                    form = aiohttp.FormData()
                    form.add_field("index_name", "reports")
                    form.add_field("file", f, filename=os.path.basename(pdf_path), content_type="application/pdf")
                    async with session.post(PDF_UPLOAD_URL, data=form) as resp:
                        resp_text = await resp.text()
                        if resp.status == 200:
                            try:
                                resp_json = json.loads(resp_text)
                                pdf_doc_id = resp_json.get("doc_id", "")
                            except Exception as e:
                                logger.error(f"[STEP 7] PDF 문서 등록 응답 JSON 파싱 실패: {e} | 본문: {resp_text}")
                                pdf_doc_id = ""
                            logger.info(f"[STEP 7] PDF 문서 등록 완료 (doc_id={pdf_doc_id})")
                        else:
                            logger.error(f"[STEP 7] PDF 문서 등록 실패: {resp_text}")
                            raise HTTPException(status_code=resp.status, detail=resp_text)
            except Exception as e:
                logger.error(f"[STEP 7] PDF 문서 등록 중 오류: {e}")
            finally:
                try:
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                        logger.info("[마무리] 임시 PDF 파일 삭제")
                except Exception as e:
                    logger.warning(f"[마무리] 임시 파일 삭제 실패: {e}")
            t_api_end = time.perf_counter()
            api_time = t_api_end - t_api_start
            total_time = t_api_end - t0
            logger.info(f"[STEP 7] 문서등록API 호출 완료 (소요={api_time:.2f}s, 누적={t_api_end-t0:.2f}s)")
            logger.info(f"[전체] 총 소요시간: {total_time:.2f}s")

            return {
                "intermediate_summary": full_summary,
                "action_items": action_items,
                "s3_report_url": download_url,
                "pdf_doc_id": pdf_doc_id
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("알 수 없는 서버 오류")
        raise HTTPException(status_code=500, detail=f"처리 중 알 수 없는 오류: {e}")

@app.get("/")
def root():
    return {"message": "회의 요약 PDF API 작동 중"}
