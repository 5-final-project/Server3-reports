FROM python:3.10-slim

# 시스템 필수 패키지 설치 (WeasyPrint 의존성 포함)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    libxml2 \
    libxslt1.1 \
    libjpeg-dev \
    fonts-nanum \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 폰트 캐시 갱신
RUN fc-cache -f -v

# 작업 디렉토리 설정
WORKDIR /app

# 로그 디렉토리 생성 및 권한 설정
RUN mkdir -p /var/logs/report_generator && \
    chmod 755 /var/logs/report_generator

# requirements.txt 먼저 복사 (Docker 캐시 최적화)
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 프로젝트 코드 복사
COPY . /app

# 환경변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8377/ || exit 1

# 포트 노출
EXPOSE 8377

# 앱 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8377", "--log-level", "info"]