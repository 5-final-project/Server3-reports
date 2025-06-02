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
          
          # 로그 디렉토리 생성 (Host에서)
          sudo mkdir -p /var/logs/report_generator
          sudo chmod 755 /var/logs/report_generator
          
          # 새 컨테이너 실행 (Filebeat 호환 경로로 볼륨 마운트)
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
            
          echo "✅ Container started with log directory mount for Filebeat"
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
          
          # JSON 로그 파일 생성 확인
          sleep 5  # 로그 파일 생성 대기
          if [ -f "/var/logs/report_generator/report_generator.log" ]; then
            echo "📝 JSON log file created successfully"
            echo "Log file size: $(du -h /var/logs/report_generator/report_generator.log 2>/dev/null || echo 'Cannot check size')"
            echo "Recent log entries:"
            tail -3 /var/logs/report_generator/report_generator.log || echo "Cannot read log file"
          else
            echo "⚠️ JSON log file not found at expected location"
          fi
        '''
      }
    }
        
    stage('ELK Integration Test') {
      steps {
        sh '''
          echo "Testing ELK Stack Integration..."
          
          # Filebeat 컨테이너 확인
          if docker ps | grep filebeat; then
            echo "✅ Filebeat container is running"
            
            # 로그 파일이 Filebeat 경로에 있는지 확인
            if [ -f "/var/logs/report_generator/report_generator.log" ]; then
              echo "✅ Log file accessible to Filebeat"
              
              # JSON 로그 형식 확인
              if head -1 /var/logs/report_generator/report_generator.log | python3 -m json.tool > /dev/null 2>&1; then
                echo "✅ Log file is in valid JSON format"
              else
                echo "⚠️ Log file may not be in valid JSON format"
              fi
            else
              echo "❌ Log file not found for Filebeat"
            fi
          else
            echo "⚠️ Filebeat container not found"
          fi
          
          # Prometheus 메트릭 확인
          if docker ps | grep team5-prom; then
            echo "✅ Team5-Prometheus container is running"
            
            # 메트릭 스크래핑 테스트
            if docker exec team5-prom wget -qO- http://${IMAGE_NAME}:8377/metrics | grep "team5_report" > /dev/null; then
              echo "✅ Prometheus can scrape Team5 report metrics"
            else
              echo "⚠️ Prometheus metrics may not be accessible"
            fi
          else
            echo "⚠️ team5-prom container not found"
          fi
          
          # Grafana 대시보드 확인
          if docker ps | grep team5-grafana; then
            echo "✅ Team5-Grafana container is running"
            echo "🎯 Import server3_6.json dashboard to view metrics"
          else
            echo "⚠️ team5-grafana container not found"
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
      sh '''
        echo "=== Final Status Check ==="
        echo "Container Status:"
        docker ps | grep ${IMAGE_NAME} || echo "Container not found"
        echo "Log Directory Status:"
        ls -la /var/logs/report_generator/ || echo "Log directory not accessible"
        echo "Recent Container Logs:"
        docker logs --tail 10 ${IMAGE_NAME} || echo "No container logs available"
      '''
    }
    success {
      echo "✅ Server3 Report Generator deployed successfully!"
      echo "🔗 Service URL: http://localhost:8377"
      echo "📊 Metrics URL: http://localhost:8377/metrics"
      echo "📝 JSON Logs: /var/logs/report_generator/report_generator.log"
      echo "🔍 Kibana: Check 'report-generator-logs-*' index"
      echo "📈 Grafana: Import server3_6.json dashboard"
      echo "🐳 Container: ${IMAGE_NAME}:${IMAGE_TAG}"
    }
    failure {
      echo "❌ Deployment failed"
      sh '''
        echo "=== Failure Analysis ==="
        echo "Container Logs:"
        docker logs ${IMAGE_NAME} || echo "No container logs available"
        echo "Container Status:"
        docker ps -a | grep ${IMAGE_NAME} || echo "Container not found"
        echo "Network Status:"
        docker network inspect team5-net | grep ${IMAGE_NAME} || echo "Not in team5-net"
        echo "Port Status:"
        netstat -tlnp | grep 8377 || echo "Port 8377 not listening"
        echo "Log Directory Permissions:"
        ls -la /var/logs/ | grep report_generator || echo "Cannot check log directory permissions"
      '''
    }
  }
}