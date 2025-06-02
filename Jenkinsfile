pipeline {
  agent { label 'team5' }
  
  environment {
    IMAGE_NAME = "server3-report"
    IMAGE_TAG = "${env.BUILD_NUMBER}"
    // Jenkins Credentials로 관리되는 환경변수들
    AWS_ACCESS_KEY_ID = credentials('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = credentials('AWS_SECRET_ACCESS_KEY')
    BUCKET_NAME = credentials('BUCKET_NAME')
    AWS_DEFAULT_REGION = credentials('AWS_DEFAULT_REGION')
  }
  
  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }
        
    stage('Build Docker Image') {
      steps {
        sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ."
      }
    }
        
    stage('Deploy Container') {
      steps {
        sh '''
          # 기존 컨테이너 정리 (이름 변경 고려)
          for name in great_villani server3-report-generator server3-report; do
            if docker ps -a --filter "name=^/${name}$" --format "{{.Names}}" | grep -q "^${name}$"; then
              echo "Stopping and removing container ${name}"
              docker rm -f "${name}"
            fi
          done
          
          # 기존 로그 디렉토리 상태 확인 (sudo 없이)
          if [ -d "/var/logs/report_generator" ]; then
            echo "Log directory already exists"
            ls -la /var/logs/report_generator/ || echo "Directory exists but cannot list"
          else
            echo "Log directory does not exist - will be created by container"
          fi
          
          # 새 컨테이너 실행 (기존 경로 그대로 유지!)
          docker run -d \\
            --name "${IMAGE_NAME}" \\
            --network team5-net \\
            --restart unless-stopped \\
            -p 8377:8377 \\
            -v /var/logs/report_generator:/var/logs/report_generator:rw \\
            -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \\
            -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \\
            -e BUCKET_NAME="${BUCKET_NAME}" \\
            -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \\
            -e ENABLE_METRICS=true \\
            -e ENVIRONMENT=production \\
            -e HOSTNAME="${IMAGE_NAME}" \\
            "${IMAGE_NAME}:${IMAGE_TAG}"
            
          echo "Container started with existing log directory mount"
        '''
      }
    }
        
    stage('Health Check') {
      steps {
        sh '''
          echo "Waiting for container to start..."
          sleep 20
          
          # 컨테이너 상태 확인
          if ! docker ps | grep ${IMAGE_NAME}; then
            echo "❌ Container not running"
            docker logs ${IMAGE_NAME} || echo "No logs available"
            exit 1
          fi
          
          # Health check with retry
          for i in 1 2 3 4 5; do
            if curl -f http://localhost:8377/ > /dev/null 2>&1; then
              echo "✅ Server3 Report Generator is running successfully"
              break
            else
              echo "Health check attempt $i/5 failed, retrying..."
              sleep 10
            fi
            
            if [ $i -eq 5 ]; then
              echo "❌ Health check failed after 5 attempts"
              docker logs ${IMAGE_NAME}
              exit 1
            fi
          done
          
          # Metrics endpoint check
          if curl -f http://localhost:8377/metrics > /dev/null 2>&1; then
            echo "📊 Metrics endpoint is working"
          else
            echo "⚠️ Metrics endpoint not available"
          fi
          
          # 로그 파일 생성 확인 (기존 경로)
          if [ -f "/var/logs/report_generator/report_generator.log" ]; then
            echo "📝 Log file is accessible"
            echo "Log file size: $(du -h /var/logs/report_generator/report_generator.log 2>/dev/null || echo 'Cannot check size')"
          else
            echo "⚠️ Log file not found at expected location"
          fi
        '''
      }
    }
        
    stage('Integration Test') {
      steps {
        sh '''
          echo "Running integration tests..."
          
          # 컨테이너 네트워크 연결 확인
          docker network inspect team5-net | grep ${IMAGE_NAME} && echo "✅ Connected to team5-net" || echo "⚠️ Network connection issue"
          
          # 컨테이너 내부에서 로그 디렉토리 확인
          docker exec ${IMAGE_NAME} ls -la /var/logs/report_generator/ && echo "✅ Log directory accessible from container" || echo "⚠️ Log directory issue"
          
          # Prometheus 메트릭 확인 (team5-prom 컨테이너가 존재하는 경우)
          if docker ps | grep team5-prom; then
            echo "Checking Prometheus metrics..."
            docker exec team5-prom wget -qO- http://${IMAGE_NAME}:8377/metrics | head -5 && echo "✅ Prometheus can scrape metrics" || echo "⚠️ Prometheus scraping issue"
          else
            echo "⚠️ team5-prom container not found, skipping metrics test"
          fi
          
          # 기존 Filebeat 연동 확인
          if docker ps | grep filebeat; then
            echo "✅ Filebeat container is running - log collection should work"
          else
            echo "⚠️ Filebeat container not found"
          fi
        '''
      }
    }
        
    stage('Cleanup') {
      steps {
        sh '''
          # 이전 이미지 정리
          docker image prune -f
          
          # 빌드 아티팩트 정리 (최근 3개 버전만 유지)
          docker images | grep ${IMAGE_NAME} | grep -v ${IMAGE_TAG} | awk '{print $3}' | head -5 | xargs -r docker rmi || echo "No old images to remove"
        '''
      }
    }
  }
  
  post {
    always {
      echo "Build #${env.BUILD_NUMBER} finished at ${new Date()}"
      // 기존 로그 디렉토리 상태 확인
      sh '''
        echo "=== Final Container Status ==="
        docker ps | grep ${IMAGE_NAME} || echo "Container not found"
        echo "=== Log Directory Status (Host) ==="
        ls -la /var/logs/report_generator/ || echo "Log directory not accessible from host"
        echo "=== Log Directory Status (Container) ==="
        docker exec ${IMAGE_NAME} ls -la /var/logs/report_generator/ || echo "Log directory not accessible from container"
        echo "=== Container Logs (last 20 lines) ==="
        docker logs --tail 20 ${IMAGE_NAME} || echo "No container logs available"
      '''
    }
    success {
      echo "✅ Server3 Report Generator deployed successfully!"
      echo "🔗 Service URL: http://localhost:8377"
      echo "📊 Metrics URL: http://localhost:8377/metrics"
      echo "📝 Logs: /var/logs/report_generator/ (기존 Filebeat 경로 유지)"
      echo "🐳 Container: ${IMAGE_NAME}:${IMAGE_TAG}"
      // 성공 알림을 위한 간단한 API 테스트
      sh '''
        echo "=== Final API Test ==="
        curl -s http://localhost:8377/ | jq . || echo "API response received"
        echo "=== Log Collection Verification ==="
        echo "Checking if logs are being written..."
        docker exec ${IMAGE_NAME} tail -5 /var/logs/report_generator/report_generator.log || echo "Cannot read recent logs"
      '''
    }
    failure {
      echo "❌ Deployment failed"
      sh '''
        echo "=== Failure Analysis ==="
        echo "Container Logs:"
        docker logs ${IMAGE_NAME} || echo "No container logs available"
        echo "=== Container Status ==="
        docker ps -a | grep ${IMAGE_NAME} || echo "Container not found"
        echo "=== Network Status ==="
        docker network inspect team5-net | grep ${IMAGE_NAME} || echo "Not in team5-net"
        echo "=== Port Status ==="
        netstat -tlnp | grep 8377 || echo "Port 8377 not listening"
        echo "=== Log Directory Permissions ==="
        ls -la /var/logs/ | grep report_generator || echo "Cannot check log directory permissions"
      '''
    }
  }
}