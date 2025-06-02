import os
import time
import logging
import re
import boto3
import markdown2
import json
import uuid
import traceback
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from weasyprint import HTML

import aiohttp
import asyncio

# ===== ELK 로깅을 위한 JSON 로거 =====
from pythonjsonlogger import jsonlogger

# ===== Prometheus 메트릭 =====
from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator


# 메트릭 정의 (실시간 성능 모니터링용)
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

# ===== ELK 최적화 JSON 로깅 설정 =====
class ELKFormatter(jsonlogger.JsonFormatter):
    """
    ELK에 최적화된 JSON 포맷터
    - Elasticsearch 인덱싱에 최적화된 필드 구조
    - Kibana 대시보드에서 필터링하기 쉬운 형태
    """
    def add_fields(self, log_record, record, message_dict):
        super(ELKFormatter, self).add_fields(log_record, record, message_dict)
        
        # ELK에서 필수적인 표준 필드들
        log_record['@timestamp'] = datetime.utcnow().isoformat() + 'Z'
        log_record['service'] = 'server3-report'
        log_record['team'] = 'team5'
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['host'] = os.getenv('HOSTNAME', 'unknown')
        log_record['environment'] = os.getenv('ENVIRONMENT', 'production')
        
        # 메시지가 dict인 경우 전개 (구조화된 로깅)
        if isinstance(record.msg, dict):
            log_record.update(record.msg)
        else:
            log_record['message'] = record.getMessage()

def setup_dual_logging():
    """
    이중 로깅 시스템 설정
    1. 콘솔: 개발/디버그용 (사람이 읽기 쉬운 형태)
    2. 파일: ELK 수집용 (JSON 구조화)
    """
    # 로그 디렉토리 생성 - 기존 Filebeat 경로와 일치
    os.makedirs("/var/logs/report_generator", exist_ok=True)  # 수정된 부분
    
    # ELK 전용 JSON 포맷터
    elk_formatter = ELKFormatter(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    # 파일 핸들러 (ELK 수집용 - JSON 형태) - 경로 수정
    file_handler = logging.FileHandler('/var/logs/report_generator/report_generator.log', encoding='utf-8')
    file_handler.setFormatter(elk_formatter)
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러 (개발/운영자용 - 읽기 쉬운 형태)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # 루트 로거 설정
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)  # ELK용
    logger.addHandler(console_handler)  # 운영자용
    
    return logger
# 로거 초기화
logger = setup_dual_logging()

# ----- 환경변수 로드 및 체크 -----
load_dotenv()
REQUIRED_ENV_VARS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "BUCKET_NAME"]
for v in REQUIRED_ENV_VARS:
    if not os.getenv(v):
        # ELK: 구조화된 에러 로그 (검색 가능)
        logger.error({
            "event": "startup_error",
            "error_type": "missing_environment_variable",
            "missing_variable": v,
            "severity": "critical",
            "action_required": "set_environment_variable"
        })
        raise RuntimeError(f"환경변수 {v}가 누락되어 있습니다.")

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
REGION = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")
PDF_UPLOAD_URL = "https://team5opensearch.ap.loclx.io/documents/upload-without-s3"
QWEN_API_URL = "https://qwen3.ap.loclx.io/api/generate"

# 서비스 시작 로그 (ELK: 서비스 상태 추적용)
logger.info({
    "event": "service_startup",
    "version": "2.7",
    "monitoring_stack": {
        "prometheus_metrics": "enabled",
        "elk_logging": "enabled",
        "grafana_dashboard": "available"
    },
    "config": {
        "aws_region": REGION,
        "bucket_name": BUCKET_NAME[:5] + "***",  # 보안상 일부만 로깅
        "pdf_upload_url": PDF_UPLOAD_URL,
        "qwen_api_url": QWEN_API_URL
    },
    "status": "started"
})

# ----- 데이터 모델 (기존 동일) -----
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

# ----- 기존 텍스트 처리 함수들 (변경 없음) -----
def add_newlines_between_sentences(text):
    # 1. 번호로 시작하는 새로운 항목 앞에 줄바꿈
    text = re.sub(r'([.?!])(\d+\.)', r'\1\n\2', text)
    # 2. 문장부호 후 띄어쓰기 없는 다음 문장에 줄바꿈
    text = re.sub(r'([.?!])([^\s\d])', r'\1\n\2', text)
    return text

def clean_llm_output(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    # <think> 태그 제거
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # 문장 줄바꿈 추가
    text = add_newlines_between_sentences(text)
    return text

# ===== 모니터링 강화된 API 호출 함수 =====
async def post_json(session: aiohttp.ClientSession, url: str, payload: dict, request_id: str = None) -> dict:
    """
    외부 API 호출 (이중 모니터링)
    - Prometheus: 호출 횟수, 응답시간 히스토그램
    - ELK: 상세한 요청/응답 내용, 에러 원인 분석
    """
    start_time = time.perf_counter()
    req_id = request_id or str(uuid.uuid4())
    
    # ELK: API 호출 상세 로그
    logger.info({
        "event": "external_api_call",
        "request_id": req_id,
        "api_url": url,
        "method": "POST",
        "payload_size_bytes": len(json.dumps(payload)),
        "api_type": "llm" if "qwen" in url.lower() else "document_upload",
        "status": "started"
    })
    
    try:
        async with session.post(url, json=payload) as response:
            resp_text = await response.text()
            elapsed = time.perf_counter() - start_time
            content_type = response.headers.get('Content-Type', '')
            
            # ELK: API 성공 로그 (응답 분석용)
            logger.info({
                "event": "external_api_call",
                "request_id": req_id,
                "api_url": url,
                "status_code": response.status,
                "response_size_bytes": len(resp_text),
                "duration_seconds": round(elapsed, 3),
                "content_type": content_type,
                "api_type": "llm" if "qwen" in url.lower() else "document_upload",
                "status": "success"
            })
            
            try:
                if "application/json" in content_type:
                    return json.loads(resp_text)
                else:
                    return json.loads(resp_text)
            except Exception as e:
                # ELK: API 파싱 에러 (디버깅용)
                logger.error({
                    "event": "external_api_call",
                    "request_id": req_id,
                    "api_url": url,
                    "error_message": str(e),
                    "error_type": "response_parse_error",
                    "response_preview": resp_text[:300],
                    "response_content_type": content_type,
                    "status": "parse_failed"
                })
                raise Exception(f"JSON 파싱 실패. 본문: {resp_text[:300]}")
                
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        # ELK: API 실패 로그 (장애 분석용)
        logger.error({
            "event": "external_api_call",
            "request_id": req_id,
            "api_url": url,
            "error_message": str(e),
            "error_type": type(e).__name__,
            "duration_seconds": round(elapsed, 3),
            "api_type": "llm" if "qwen" in url.lower() else "document_upload",
            "status": "failed"
        })
        raise

# ===== 모니터링 강화된 LLM 함수들 =====
async def async_llm_map_summary(chunk_text: str, session: aiohttp.ClientSession, request_id: str = None) -> str:
    payload = {
        "prompt": [{"role": "user", "content": f"아래 회의내용을 핵심을 중심으로 요약해주세요. \n핵심내용에 번호를 순서대로 붙여주세요. \n마지막에 추가 요약은 절대 하지 마세요.\n\n회의내용:\n{chunk_text}"}],
        "max_tokens": 30000,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    # ELK: LLM 작업 추적 (비즈니스 로직 분석용)
    logger.info({
        "event": "llm_operation",
        "request_id": request_id,
        "operation_type": "map_summary",
        "input_length": len(chunk_text),
        "llm_params": {
            "max_tokens": payload["max_tokens"],
            "temperature": payload["temperature"],
            "top_p": payload["top_p"]
        },
        "status": "started"
    })
    
    try:
        result = await post_json(session, QWEN_API_URL, payload, request_id)
        response_text = clean_llm_output(result.get("response", ""))
        
        # ELK: LLM 성공 로그 (품질 분석용)
        logger.info({
            "event": "llm_operation",
            "request_id": request_id,
            "operation_type": "map_summary",
            "input_length": len(chunk_text),
            "output_length": len(response_text),
            "compression_ratio": round(len(response_text) / len(chunk_text), 2),
            "status": "completed"
        })
        
        return response_text
    except Exception as e:
        # ELK: LLM 실패 로그 (에러 분석용)
        logger.error({
            "event": "llm_operation",
            "request_id": request_id,
            "operation_type": "map_summary",
            "input_length": len(chunk_text),
            "error_message": str(e),
            "error_type": type(e).__name__,
            "status": "failed"
        })
        raise

async def async_llm_combine_summary(map_summaries: list, session: aiohttp.ClientSession, request_id: str = None) -> str:
    combined_text = "\n".join(map_summaries)
    payload = {
        "prompt": [{"role": "user", "content": f"여러 회의 요약문이 아래에 나열되어 있습니다. 중복 없이 통합 정리해주세요. \n핵심내용에 번호를 순서대로 붙여주세요. \n마지막에 추가 요약은 절대 하지 마세요.\n\n{combined_text}"}],
        "max_tokens": 30000,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    # ELK: LLM 결합 작업 로그
    logger.info({
        "event": "llm_operation",
        "request_id": request_id,
        "operation_type": "combine_summary",
        "input_summaries_count": len(map_summaries),
        "combined_length": len(combined_text),
        "status": "started"
    })
    
    try:
        result = await post_json(session, QWEN_API_URL, payload, request_id)
        response_text = clean_llm_output(result.get("response", ""))
        
        logger.info({
            "event": "llm_operation",
            "request_id": request_id,
            "operation_type": "combine_summary",
            "input_summaries_count": len(map_summaries),
            "output_length": len(response_text),
            "status": "completed"
        })
        
        return response_text
    except Exception as e:
        logger.error({
            "event": "llm_operation",
            "request_id": request_id,
            "operation_type": "combine_summary",
            "error_message": str(e),
            "error_type": type(e).__name__,
            "status": "failed"
        })
        raise

async def async_llm_map_action_items(summary_text: str, session: aiohttp.ClientSession, request_id: str = None) -> str:
    payload = {
        "prompt": [{"role": "user", "content": f"아래 요약문을 바탕으로 앞으로 해야할 일(To-Do List)을 작성해주세요. 각 list 항목에 번호를 붙여주세요. \n 마지막에 추가 요약은 절대 하지 마세요.\n\n{summary_text}"}],
        "max_tokens": 30000,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    logger.info({
        "event": "llm_operation",
        "request_id": request_id,
        "operation_type": "map_action_items",
        "input_length": len(summary_text),
        "status": "started"
    })
    
    try:
        result = await post_json(session, QWEN_API_URL, payload, request_id)
        response_text = clean_llm_output(result.get("response", ""))
        
        logger.info({
            "event": "llm_operation",
            "request_id": request_id,
            "operation_type": "map_action_items",
            "output_length": len(response_text),
            "status": "completed"
        })
        
        return response_text
    except Exception as e:
        logger.error({
            "event": "llm_operation",
            "request_id": request_id,
            "operation_type": "map_action_items",
            "error_message": str(e),
            "error_type": type(e).__name__,
            "status": "failed"
        })
        raise

async def async_llm_combine_action_items(action_items_list: list, session: aiohttp.ClientSession, request_id: str = None) -> str:
    combined_text = "\n".join(action_items_list)
    payload = {
        "prompt": [{"role": "user", "content": f"앞으로 해야할 일들의 리스트가 아래에 나열되어 있습니다. 중복을 제거하고, 통합 정리해서 To-Do List만 번호와 함께 다시 출력해주세요. 마지막에 추가 요약은 하지 마세요.\n\n{combined_text}"}],
        "max_tokens": 30000,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    logger.info({
        "event": "llm_operation",
        "request_id": request_id,
        "operation_type": "combine_action_items",
        "input_count": len(action_items_list),
        "combined_length": len(combined_text),
        "status": "started"
    })
    
    try:
        result = await post_json(session, QWEN_API_URL, payload, request_id)
        response_text = clean_llm_output(result.get("response", ""))
        
        logger.info({
            "event": "llm_operation",
            "request_id": request_id,
            "operation_type": "combine_action_items",
            "output_length": len(response_text),
            "status": "completed"
        })
        
        return response_text
    except Exception as e:
        logger.error({
            "event": "llm_operation",
            "request_id": request_id,
            "operation_type": "combine_action_items",
            "error_message": str(e),
            "error_type": type(e).__name__,
            "status": "failed"
        })
        raise

# ----- 기존 보고서 생성 함수들 (변경 없음) -----
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

def html_to_pdf(html, pdf_path, request_id: str = None):
    """
    PDF 생성 (이중 모니터링)
    - Prometheus: 생성 시간 히스토그램, 생성 횟수
    - ELK: 파일 크기, 에러 원인, 생성 과정 추적
    """
    start_time = time.perf_counter()
    
    # ELK: PDF 생성 로그
    logger.info({
        "event": "pdf_generation",
        "request_id": request_id,
        "pdf_path": pdf_path,
        "html_size_bytes": len(html),
        "status": "started"
    })
    
    try:
        HTML(string=html).write_pdf(pdf_path)
        elapsed = time.perf_counter() - start_time
        file_size = os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0
        
        # ELK: PDF 생성 성공 로그
        logger.info({
            "event": "pdf_generation",
            "request_id": request_id,
            "pdf_path": pdf_path,
            "file_size_bytes": file_size,
            "duration_seconds": round(elapsed, 3),
            "html_to_pdf_ratio": round(file_size / len(html), 2) if len(html) > 0 else 0,
            "status": "completed"
        })
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        # ELK: PDF 생성 실패 로그
        logger.error({
            "event": "pdf_generation",
            "request_id": request_id,
            "pdf_path": pdf_path,
            "html_size_bytes": len(html),
            "error_message": str(e),
            "error_type": type(e).__name__,
            "duration_seconds": round(elapsed, 3),
            "status": "failed"
        })
        raise

def upload_to_s3(file_path, bucket, key, request_id: str = None):
    """
    S3 업로드 (이중 모니터링)
    - Prometheus: 업로드 시간, 업로드 횟수
    - ELK: 파일 정보, S3 응답, 에러 분석
    """
    start_time = time.perf_counter()
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    
    # ELK: S3 업로드 로그
    logger.info({
        "event": "s3_upload",
        "request_id": request_id,
        "file_path": file_path,
        "bucket": bucket,
        "s3_key": key,
        "file_size_bytes": file_size,
        "aws_region": REGION,
        "status": "started"
    })
    
    try:
        s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=REGION)
        s3.upload_file(file_path, bucket, key)
        url = s3.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=3600)
        elapsed = time.perf_counter() - start_time
        
        # ELK: S3 업로드 성공 로그
        logger.info({
            "event": "s3_upload",
            "request_id": request_id,
            "bucket": bucket,
            "s3_key": key,
            "file_size_bytes": file_size,
            "duration_seconds": round(elapsed, 3),
            "upload_speed_mbps": round((file_size / 1024 / 1024) / elapsed, 2) if elapsed > 0 else 0,
            "presigned_url_expires_seconds": 3600,
            "status": "completed"
        })
        
        return url
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        # ELK: S3 업로드 실패 로그
        logger.error({
            "event": "s3_upload",
            "request_id": request_id,
            "bucket": bucket,
            "s3_key": key,
            "file_size_bytes": file_size,
            "aws_region": REGION,
            "error_message": str(e),
            "error_type": type(e).__name__,
            "duration_seconds": round(elapsed, 3),
            "status": "failed"
        })
        raise

# ----- FastAPI 앱 -----
app = FastAPI(title="회의 요약 PDF API", version="2.7")

# Prometheus 계측 추가 (실시간 메트릭용)
Instrumentator().instrument(app).expose(app)

@app.middleware("http")
async def dual_monitoring_middleware(request: Request, call_next):
    """
    이중 모니터링 미들웨어
    - Prometheus: 자동 HTTP 메트릭 (Instrumentator가 처리)
    - ELK: 상세한 요청/응답 로그
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # ELK: HTTP 요청 시작 로그 (사용자 행동 분석용)
    logger.info({
        "event": "http_request",
        "request_id": request_id,
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "query_params": dict(request.query_params),
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "content_type": request.headers.get("content-type"),
        "content_length": request.headers.get("content-length"),
        "referer": request.headers.get("referer"),
        "status": "started"
    })
    
    # request_id를 요청 상태에 저장
    request.state.request_id = request_id
    
    response = await call_next(request)
    
    # ELK: HTTP 요청 완료 로그 (성능 분석용)
    duration = time.time() - start_time
    logger.info({
        "event": "http_request",
        "request_id": request_id,
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_seconds": round(duration, 3),
        "response_size": response.headers.get("content-length"),
        "status": "completed"
    })
    
    return response

@app.post("/report-json")
async def report_json(request: MeetingInput, req: Request):
    """
    회의 요약 보고서 생성 API
    
    모니터링 전략:
    - Prometheus: 전체 처리량, 에러율, 성능 트렌드
    - ELK: 개별 요청 추적, 에러 상세 분석, 사용 패턴
    """
    # 미들웨어에서 설정한 request_id 가져오기
    request_id = getattr(req.state, 'request_id', str(uuid.uuid4()))
    
    # Prometheus: 실시간 메트릭 업데이트
    team5_report_requests.labels(service="server3-report").inc()
    team5_active_requests.labels(service="server3-report").inc()
    
    # ELK: 보고서 생성 요청 분석 로그
    logger.info({
        "event": "report_generation_request",
        "request_id": request_id,
        "input_text_length": len(request.text_stt),
        "has_meeting_meta": request.meeting_meta is not None,
        "meeting_title": request.meeting_meta.title if request.meeting_meta else None,
        "meeting_author": request.meeting_meta.author if request.meeting_meta else None,
        "meeting_participants_count": len(request.meeting_meta.participants) if request.meeting_meta and request.meeting_meta.participants else 0,
        "meeting_datetime": request.meeting_meta.datetime if request.meeting_meta else None,
        "status": "started"
    })
    
    try:
        t0 = time.perf_counter()

        # Step 1. 텍스트 분할
        t_split_start = time.perf_counter()
        splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=100)
        chunks = splitter.split_text(request.text_stt)
        t_split_end = time.perf_counter()
        split_time = t_split_end - t_split_start
        
        # ELK: 텍스트 분할 분석 로그
        logger.info({
            "event": "text_processing",
            "request_id": request_id,
            "operation": "chunking",
            "input_length": len(request.text_stt),
            "chunks_count": len(chunks),
            "average_chunk_size": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
            "chunk_sizes": [len(chunk) for chunk in chunks],
            "duration_seconds": round(split_time, 3),
            "status": "completed"
        })
        
        if not chunks:
            # Prometheus: 에러 카운트
            team5_report_errors.labels(service="server3-report").inc()
            
            # ELK: 빈 청크 에러 로그
            logger.error({
                "event": "report_generation_request",
                "request_id": request_id,
                "error_message": "텍스트 분할 결과 빈 청크",
                "error_type": "empty_chunks_after_split",
                "input_text_preview": request.text_stt[:200],
                "status": "failed"
            })
            raise HTTPException(status_code=400, detail="요약할 텍스트가 없습니다.")

        async with aiohttp.ClientSession() as session:
            # Step 2. Map 요약
            t_map_start = time.perf_counter()
            try:
                # Prometheus: LLM 호출 메트릭
                team5_llm_calls.labels(service="server3-report", operation="map_summary").inc()
                with team5_llm_call_duration.labels(service="server3-report", operation="map_summary").time():
                    map_summaries = await asyncio.gather(*[async_llm_map_summary(c, session, request_id) for c in chunks])
            except Exception as e:
                # ELK: Map 요약 실패 로그
                logger.error({
                    "event": "report_generation_step",
                    "request_id": request_id,
                    "step": "map_summary",
                    "chunks_count": len(chunks),
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                    "status": "failed"
                })
                team5_report_errors.labels(service="server3-report").inc()
                raise HTTPException(status_code=500, detail=f"map 요약 실패: {e}")
            t_map_end = time.perf_counter()
            map_time = t_map_end - t_map_start

            # Step 3. Combine 요약
            t_combine_start = time.perf_counter()
            try:
                if len(map_summaries) == 1:
                    full_summary = map_summaries[0]
                    
                    # ELK: 단일 요약 로그
                    logger.info({
                        "event": "report_generation_step",
                        "request_id": request_id,
                        "step": "combine_summary",
                        "action": "skipped_single_summary",
                        "summary_length": len(full_summary),
                        "status": "completed"
                    })
                else:
                    team5_llm_calls.labels(service="server3-report", operation="combine_summary").inc()
                    with team5_llm_call_duration.labels(service="server3-report", operation="combine_summary").time():
                        full_summary = await async_llm_combine_summary(map_summaries, session, request_id)
            except Exception as e:
                logger.error({
                    "event": "report_generation_step",
                    "request_id": request_id,
                    "step": "combine_summary",
                    "map_summaries_count": len(map_summaries),
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                    "status": "failed"
                })
                team5_report_errors.labels(service="server3-report").inc()
                raise HTTPException(status_code=500, detail=f"combine 요약 실패: {e}")
            t_combine_end = time.perf_counter()
            combine_time = t_combine_end - t_combine_start

            # Step 4. Action Items - Map-Reduce 방식
            t_action_start = time.perf_counter()
            try:
                team5_llm_calls.labels(service="server3-report", operation="action_items").inc()
                with team5_llm_call_duration.labels(service="server3-report", operation="action_items").time():
                    # 4-1. 각 chunk 요약별로 action item 추출
                    map_action_items = await asyncio.gather(
                        *[async_llm_map_action_items(summary, session, request_id) for summary in map_summaries]
                    )
                    # 4-2. 여러 action item을 LLM으로 통합
                    if len(map_action_items) == 1:
                        action_items = map_action_items[0]
                    else:
                        action_items = await async_llm_combine_action_items(map_action_items, session, request_id)
            except Exception as e:
                logger.error({
                    "event": "report_generation_step",
                    "request_id": request_id,
                    "step": "action_items",
                    "map_summaries_count": len(map_summaries),
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                    "status": "failed"
                })
                team5_report_errors.labels(service="server3-report").inc()
                raise HTTPException(status_code=500, detail=f"action item(map-reduce) 생성 실패: {e}")
            t_action_end = time.perf_counter()
            action_time = t_action_end - t_action_start

            # Step 5. PDF 생성
            ts = int(time.time())
            pdf_path = os.path.abspath(f"meeting_report_{ts}.pdf")
            md = render_markdown_report(full_summary, action_items)
            html = markdown_to_html(md)
            t_pdf_start = time.perf_counter()
            try:
                # Prometheus: PDF 생성 메트릭
                with team5_pdf_generation_duration.labels(service="server3-report").time():
                    html_to_pdf(html, pdf_path, request_id)
                team5_pdf_generations.labels(service="server3-report").inc()
            except Exception as e:
                logger.error({
                    "event": "report_generation_step",
                    "request_id": request_id,
                    "step": "pdf_generation",
                    "html_size_bytes": len(html),
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                    "status": "failed"
                })
                team5_report_errors.labels(service="server3-report").inc()
                raise HTTPException(status_code=500, detail=f"PDF 생성 실패: {e}")
            t_pdf_end = time.perf_counter()
            pdf_time = t_pdf_end - t_pdf_start

            # Step 6. S3 업로드
            t_s3_start = time.perf_counter()
            try:
                s3_key = f"reports/meeting_report_{ts}.pdf"
                # Prometheus: S3 업로드 메트릭
                with team5_s3_upload_duration.labels(service="server3-report").time():
                    download_url = upload_to_s3(pdf_path, BUCKET_NAME, s3_key, request_id)
                team5_s3_uploads.labels(service="server3-report").inc()
            except Exception as e:
                logger.error({
                    "event": "report_generation_step",
                    "request_id": request_id,
                    "step": "s3_upload",
                    "s3_key": s3_key,
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                    "status": "failed"
                })
                team5_report_errors.labels(service="server3-report").inc()
                raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {e}")
            t_s3_end = time.perf_counter()
            s3_time = t_s3_end - t_s3_start

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
                                
                                # ELK: 문서 등록 성공 로그
                                logger.info({
                                    "event": "document_registration",
                                    "request_id": request_id,
                                    "doc_id": pdf_doc_id,
                                    "index_name": "reports",
                                    "file_name": os.path.basename(pdf_path),
                                    "status": "completed"
                                })
                                
                            except Exception as e:
                                logger.error({
                                    "event": "document_registration",
                                    "request_id": request_id,
                                    "error_message": f"JSON 파싱 실패: {str(e)}",
                                    "response_preview": resp_text[:200],
                                    "status": "parse_failed"
                                })
                                pdf_doc_id = ""
                        else:
                            logger.error({
                                "event": "document_registration",
                                "request_id": request_id,
                                "status_code": resp.status,
                                "error_message": resp_text,
                                "status": "http_error"
                            })
                            raise HTTPException(status_code=resp.status, detail=resp_text)
            except Exception as e:
                logger.error({
                    "event": "document_registration",
                    "request_id": request_id,
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                    "status": "exception"
                })
            finally:
                try:
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                        logger.info({
                            "event": "file_cleanup",
                            "request_id": request_id,
                            "file_path": pdf_path,
                            "action": "deleted",
                            "status": "completed"
                        })
                except Exception as e:
                    logger.warning({
                        "event": "file_cleanup",
                        "request_id": request_id,
                        "file_path": pdf_path,
                        "error_message": str(e),
                        "status": "failed"
                    })
            t_api_end = time.perf_counter()
            api_time = t_api_end - t_api_start
            total_time = t_api_end - t0

            # ELK: 전체 보고서 생성 완료 로그 (성능 분석용)
            logger.info({
                "event": "report_generation_request",
                "request_id": request_id,
                "total_duration_seconds": round(total_time, 3),
                "step_durations": {
                    "text_chunking": round(split_time, 3),
                    "map_summary": round(map_time, 3),
                    "combine_summary": round(combine_time, 3),
                    "action_items": round(action_time, 3),
                    "pdf_generation": round(pdf_time, 3),
                    "s3_upload": round(s3_time, 3),
                    "document_registration": round(api_time, 3)
                },
                "processing_stats": {
                    "input_length": len(request.text_stt),
                    "chunks_count": len(chunks),
                    "summary_length": len(full_summary),
                    "action_items_length": len(action_items),
                    "pdf_doc_id": pdf_doc_id,
                    "s3_url_provided": bool(download_url)
                },
                "performance_ratios": {
                    "chars_per_second": round(len(request.text_stt) / total_time, 2),
                    "chunks_per_second": round(len(chunks) / total_time, 2)
                },
                "status": "completed"
            })

            # Prometheus: 전체 처리 시간 메트릭
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
        # ELK: 예상치 못한 에러 로그 (상세 디버깅용)
        logger.error({
            "event": "report_generation_request",
            "request_id": request_id,
            "error_message": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "input_preview": request.text_stt[:200] if hasattr(request, 'text_stt') else None,
            "status": "unexpected_error"
        })
        # Prometheus: 에러 카운트
        team5_report_errors.labels(service="server3-report").inc()
        raise HTTPException(status_code=500, detail=f"처리 중 알 수 없는 오류: {e}")
    finally:
        # Prometheus: 활성 요청 수 감소
        team5_active_requests.labels(service="server3-report").dec()

@app.get("/")
def root():
    return {"message": "회의 요약 PDF API 작동 중"}

# 서비스 종료 시 로그
@app.on_event("shutdown")
async def shutdown_event():
    logger.info({
        "event": "service_shutdown",
        "status": "graceful_shutdown",
        "timestamp": datetime.utcnow().isoformat() + 'Z'
    })