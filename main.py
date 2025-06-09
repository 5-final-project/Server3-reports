import os
import time
import logging
import re
import boto3
import markdown2
import json
import requests
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from weasyprint import HTML

import aiohttp
import asyncio

# ===== JSON 로깅 설정 추가 =====
def setup_json_logging():
    """JSON 형태로 로그를 파일에 기록하는 핸들러 설정 (ELK 스택 연동)"""
    
    # 로그 디렉토리 확인/생성
    log_dir = "/var/logs/report_generator"
    os.makedirs(log_dir, exist_ok=True)
    
    # JSON 포맷터 클래스
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "@timestamp": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname,
                "service": "server3-report",
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "logger": record.name
            }
            
            # 예외 정보가 있으면 추가
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)
            
            # 추가 필드가 있으면 포함
            if hasattr(record, 'extra_fields'):
                log_entry.update(record.extra_fields)
            
            return json.dumps(log_entry, ensure_ascii=False)
    
    # 파일 핸들러 생성
    log_file_path = f"{log_dir}/report_generator.log"
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(JsonFormatter())
    file_handler.setLevel(logging.INFO)
    
    # 루트 로거에 핸들러 추가
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    print(f"✅ JSON logging setup completed: {log_file_path}")
    return log_file_path

# JSON 로깅 설정 (앱 생성 전에 실행)
log_file_path = setup_json_logging()

# ===== Prometheus 메트릭 추가 =====
from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

# 메트릭 정의
team5_report_requests = Counter('team5_report_requests_total', 'Total report requests', ['service'])
team5_llm_calls = Counter('team5_llm_calls_total', 'Total LLM calls', ['service', 'operation'])
team5_llm_call_duration = Histogram('team5_llm_call_seconds', 'LLM call duration', ['service', 'operation'])
team5_report_errors = Counter('team5_report_errors_total', 'Total report errors', ['service'])
team5_active_requests = Gauge('team5_report_active_requests', 'Active report requests', ['service'])
team5_pdf_generations = Counter('team5_pdf_generations_total', 'Total PDF generations', ['service'])
team5_s3_uploads = Counter('team5_s3_uploads_total', 'Total S3 uploads', ['service'])
team5_pdf_generation_duration = Histogram('team5_pdf_generation_seconds', 'PDF generation duration', ['service'])
team5_s3_upload_duration = Histogram('team5_s3_upload_seconds', 'S3 upload duration', ['service'])
team5_report_processing_duration = Histogram('team5_report_processing_seconds', 'Total report processing time', ['service'])

# ----- 기존 로그 설정 (JSON 로그와 함께 사용) -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("meeting-summary")

# 로그 시작 메시지
logger.info("Server3-Report starting with JSON logging enabled", extra={
    'extra_fields': {
        'event': 'service_startup',
        'log_file': log_file_path,
        'version': '2.8'
    }
})

# ----- 환경변수 로드 및 체크 (RunPod 설정 추가) -----
load_dotenv()
REQUIRED_ENV_VARS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "BUCKET_NAME", "RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID"]
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

# RunPod 설정
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

# ----- 기존 데이터 모델 (변경 없음) -----
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

# ----- 기존 줄바꿈 후처리 함수 (변경 없음) -----
def add_newlines_between_sentences(text):
    # 1. 번호로 시작하는 새로운 항목 앞에 줄바꿈
    text = re.sub(r'([.?!])(\d+\.)', r'\1\n\2', text)
    # 2. 문장부호 후 띄어쓰기 없는 다음 문장에 줄바꿈
    text = re.sub(r'([.?!])([^\s\d])', r'\1\n\2', text)
    return text

# ----- 기존 LLM 처리 함수 (변경 없음) -----
def clean_llm_output(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    # <think> 태그 제거
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # 문장 줄바꿈 추가
    text = add_newlines_between_sentences(text)
    return text

# ----- RunPod API 함수 추가 -----
def run_llm_job_sync(prompt: str, max_tokens: int = 8192, temperature: float = 0.7):
    """RunPod runsync 요청을 보내고, 결과가 완료될 때까지 기다린 후 출력."""
    # 헤더 설정
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {RUNPOD_API_KEY}'
    }
    
    # 요청 본문
    payload = {
        'input': {
            'prompt': prompt,
            'sampling_params': {
                'max_tokens': max_tokens,
                'temperature': temperature,
                'top_p': 0.95
            }
        }
    }
    
    # POST 요청 (runsync)
    logger.info("RunPod LLM 요청 전송 중...")
    response = requests.post(
        f'https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync',
        headers=headers,
        json=payload,
        timeout=300  # 5분 타임아웃
    )
    
    if response.status_code != 200:
        raise Exception(f"RunPod API 요청 실패: {response.status_code} - {response.text}")
    
    res_json = response.json()
    job_id = res_json.get("id")
    status = res_json.get("status")
    
    # 만약 바로 결과가 오면 처리
    if status == "COMPLETED":
        logger.info("RunPod 작업 완료")
        return res_json['output']
    
    # 진행 중이면 폴링 시작
    logger.info(f"RunPod 작업 진행 (job_id: {job_id})")
    max_retries = 150  # 최대 5분 대기 (2초 * 150)
    retry_count = 0
    
    while retry_count < max_retries:
        time.sleep(2)  # 2초 간격으로 조회
        try:
            poll_response = requests.get(
                f'https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}',
                headers=headers,
                timeout=30
            )
            
            if poll_response.status_code != 200:
                logger.warning(f"RunPod 상태 조회 실패: {poll_response.status_code}")
                retry_count += 1
                continue
                
            poll_json = poll_response.json()
            poll_status = poll_json.get("status")
            
            if poll_status == "COMPLETED":
                logger.info("RunPod 작업 완료")
                return poll_json['output']
            elif poll_status == "FAILED":
                error_msg = poll_json.get('error', 'No error message')
                logger.error(f"RunPod 작업 실패: {error_msg}")
                raise Exception(f"RunPod 작업 실패: {error_msg}")
            elif poll_status == "CANCELLED":
                logger.error("RunPod 작업 취소")
                raise Exception("RunPod 작업 취소")
            elif poll_status == "IN_PROGRESS":
                logger.info("RunPod 작업 진행 중")
            else:
                logger.info(f"RunPod 작업 대기 중 (status: {poll_status})")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"RunPod 상태 조회 중 네트워크 오류: {e}")
        
        retry_count += 1
    
    # 타임아웃
    raise Exception("RunPod 작업 타임아웃 (5분 초과)")

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

# RunPod를 사용하는 combine 요약 함수로 수정
def llm_combine_summary_runpod(map_summaries: list) -> str:
    """RunPod API를 사용하여 여러 요약을 통합"""
    combined_text = "\n".join(map_summaries)
    prompt = f"""여러 회의 요약문이 아래에 나열되어 있습니다. 중복 없이 통합 정리해주세요. 
핵심내용에 번호를 순서대로 붙여주세요. 
마지막에 추가 요약은 절대 하지 마세요.

{combined_text}"""
    
    try:
        result = run_llm_job_sync(prompt, max_tokens=8192, temperature=0.7)
        return clean_llm_output(result)
    except Exception as e:
        logger.error(f"llm_combine_summary_runpod 에러: {e}")
        raise

# [기존] Action Item map-reduce 방식 함수들 (변경 없음)
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

# RunPod를 사용하는 Action Items combine 함수로 수정
def llm_combine_action_items_runpod(action_items_list: list) -> str:
    """RunPod API를 사용하여 여러 Action Items를 통합"""
    combined_text = "\n".join(action_items_list)
    prompt = f"""앞으로 해야할 일들의 리스트가 아래에 나열되어 있습니다. 중복을 제거하고, 통합 정리해서 To-Do List만 번호와 함께 다시 출력해주세요. 마지막에 추가 요약은 하지 마세요.

{combined_text}"""
    
    try:
        result = run_llm_job_sync(prompt, max_tokens=8192, temperature=0.7)
        return clean_llm_output(result)
    except Exception as e:
        logger.error(f"llm_combine_action_items_runpod 에러: {e}")
        raise

# ----- 기존 보고서 생성 함수 (변경 없음) -----
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

# ----- FastAPI 앱 (메트릭 추가) -----
app = FastAPI(title="회의 요약 PDF API", version="2.8")

# Prometheus 계측 추가
Instrumentator().instrument(app).expose(app)

@app.post("/report-json")
async def report_json(request: MeetingInput):
    # 메트릭: 요청 시작
    team5_report_requests.labels(service="server3-report").inc()
    team5_active_requests.labels(service="server3-report").inc()
    
    try:
        t0 = time.perf_counter()
        logger.info("요약 요청 수신", extra={
            'extra_fields': {
                'event': 'report_request_received',
                'text_length': len(request.text_stt)
            }
        })

        # Step 1. 텍스트 분할 (기존 로직 동일)
        t_split_start = time.perf_counter()
        splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=100)
        chunks = splitter.split_text(request.text_stt)
        t_split_end = time.perf_counter()
        split_time = t_split_end - t_split_start
        logger.info(f"[STEP 1] 텍스트 분할 완료 (chunk={len(chunks)}, 소요={split_time:.2f}s, 누적={t_split_end-t0:.2f}s)")
        if not chunks:
            team5_report_errors.labels(service="server3-report").inc()  # 메트릭 추가
            raise HTTPException(status_code=400, detail="요약할 텍스트가 없습니다.")

        async with aiohttp.ClientSession() as session:
            # Step 2. Map 요약 (기존 로직 + 메트릭)
            t_map_start = time.perf_counter()
            try:
                # 메트릭: LLM 호출 카운트 및 시간 측정
                team5_llm_calls.labels(service="server3-report", operation="map_summary").inc()
                with team5_llm_call_duration.labels(service="server3-report", operation="map_summary").time():
                    map_summaries = await asyncio.gather(*[async_llm_map_summary(c, session) for c in chunks])
            except Exception as e:
                logger.error(f"[STEP 2] map 요약 실패: {e}")
                team5_report_errors.labels(service="server3-report").inc()  # 메트릭 추가
                raise HTTPException(status_code=500, detail=f"map 요약 실패: {e}")
            t_map_end = time.perf_counter()
            map_time = t_map_end - t_map_start
            logger.info(f"[STEP 2] map 요약 완료 (소요={map_time:.2f}s, 누적={t_map_end-t0:.2f}s)")

            # Step 3. Combine 요약 (RunPod API 사용으로 수정)
            t_combine_start = time.perf_counter()
            try:
                if len(map_summaries) == 1:
                    full_summary = map_summaries[0]
                else:
                    team5_llm_calls.labels(service="server3-report", operation="combine_summary_runpod").inc()
                    with team5_llm_call_duration.labels(service="server3-report", operation="combine_summary_runpod").time():
                        full_summary = llm_combine_summary_runpod(map_summaries)
            except Exception as e:
                logger.error(f"[STEP 3] combine 요약(RunPod) 실패: {e}")
                team5_report_errors.labels(service="server3-report").inc()  # 메트릭 추가
                raise HTTPException(status_code=500, detail=f"combine 요약(RunPod) 실패: {e}")
            t_combine_end = time.perf_counter()
            combine_time = t_combine_end - t_combine_start
            logger.info(f"[STEP 3] combine 요약(RunPod) 완료 (소요={combine_time:.2f}s, 누적={t_combine_end-t0:.2f}s)")

            # Step 4. Action Items - Map-Reduce 방식 (일부 RunPod 사용으로 수정)
            t_action_start = time.perf_counter()
            try:
                team5_llm_calls.labels(service="server3-report", operation="action_items").inc()
                with team5_llm_call_duration.labels(service="server3-report", operation="action_items").time():
                    # 4-1. 각 chunk 요약별로 action item 추출 (기존 API 사용)
                    map_action_items = await asyncio.gather(
                        *[async_llm_map_action_items(summary, session) for summary in map_summaries]
                    )
                    # 4-2. 여러 action item을 RunPod로 통합
                    if len(map_action_items) == 1:
                        action_items = map_action_items[0]
                    else:
                        action_items = llm_combine_action_items_runpod(map_action_items)
            except Exception as e:
                logger.error(f"[STEP 4] action item(map-reduce with RunPod) 생성 실패: {e}")
                team5_report_errors.labels(service="server3-report").inc()  # 메트릭 추가
                raise HTTPException(status_code=500, detail=f"action item(map-reduce with RunPod) 생성 실패: {e}")
            t_action_end = time.perf_counter()
            action_time = t_action_end - t_action_start
            logger.info(f"[STEP 4] action items(map-reduce with RunPod) 생성 완료 (소요={action_time:.2f}s, 누적={t_action_end-t0:.2f}s)")

            # Step 5. PDF 생성 (기존 로직 + 메트릭)
            ts = int(time.time())
            pdf_path = os.path.abspath(f"meeting_report_{ts}.pdf")
            md = render_markdown_report(full_summary, action_items)
            html = markdown_to_html(md)
            t_pdf_start = time.perf_counter()
            try:
                with team5_pdf_generation_duration.labels(service="server3-report").time():
                    html_to_pdf(html, pdf_path)
                team5_pdf_generations.labels(service="server3-report").inc()  # 메트릭 추가
            except Exception as e:
                logger.error(f"[STEP 5] PDF 생성 실패: {e}")
                team5_report_errors.labels(service="server3-report").inc()  # 메트릭 추가
                raise HTTPException(status_code=500, detail=f"PDF 생성 실패: {e}")
            t_pdf_end = time.perf_counter()
            pdf_time = t_pdf_end - t_pdf_start
            logger.info(f"[STEP 5] PDF 생성 완료 (소요={pdf_time:.2f}s, 누적={t_pdf_end-t0:.2f}s)")

            # Step 6. S3 업로드 (기존 로직 + 메트릭)
            t_s3_start = time.perf_counter()
            try:
                s3_key = f"reports/meeting_report_{ts}.pdf"
                with team5_s3_upload_duration.labels(service="server3-report").time():
                    download_url = upload_to_s3(pdf_path, BUCKET_NAME, s3_key)
                team5_s3_uploads.labels(service="server3-report").inc()  # 메트릭 추가
            except Exception as e:
                logger.error(f"[STEP 6] S3 업로드 실패: {e}")
                team5_report_errors.labels(service="server3-report").inc()  # 메트릭 추가
                raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {e}")
            t_s3_end = time.perf_counter()
            s3_time = t_s3_end - t_s3_start
            logger.info(f"[STEP 6] S3 업로드 완료 (소요={s3_time:.2f}s, 누적={t_s3_end-t0:.2f}s)")

            # Step 7. PDF 문서 등록 API 호출 (기존 로직 동일)
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
            logger.info(f"[전체] 총 소요시간: {total_time:.2f}s", extra={
                'extra_fields': {
                    'event': 'report_completed',
                    'total_time': total_time,
                    'pdf_doc_id': pdf_doc_id
                }
            })

            # 메트릭: 전체 처리 시간 기록
            team5_report_processing_duration.labels(service="server3-report").observe(total_time)

            return {
                "intermediate_summary": full_summary,
                "action_items": action_items,
                "s3_report_url": download_url,
                "pdf_doc_id": pdf_doc_id
            }

    except HTTPException:
        raise
    except Exception as e:
        team5_report_errors.labels(service="server3-report").inc()  # 메트릭 추가
        logger.exception("알 수 없는 서버 오류", extra={
            'extra_fields': {
                'event': 'unexpected_error',
                'error_type': type(e).__name__
            }
        })
        raise HTTPException(status_code=500, detail=f"처리 중 알 수 없는 오류: {e}")
    finally:
        # 메트릭: 활성 요청 수 감소
        team5_active_requests.labels(service="server3-report").dec()

@app.get("/")
def root():
    logger.info("Root endpoint accessed", extra={
        'extra_fields': {
            'event': 'root_endpoint_access',
            'timestamp': datetime.utcnow().isoformat()
        }
    })
    return {"message": "회의 요약 PDF API 작동 중", "version": "2.7", "logging": "enabled"}

# 추가: 헬스체크 엔드포인트
@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "version": "2.7", 
        "logging": "json_enabled",
        "log_file": log_file_path
    }

# 추가: /metrics 엔드포인트는 Instrumentator가 자동 생성