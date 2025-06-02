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
          # 기존 컨테이너 정리 (sh 호환 문법)
          for name in great_villani server3-report-generator server3-report; do
            if docker ps -a --filter "name=^/${name}$" --format "{{.Names}}" | grep -q "^${name}$"; then
              echo "Stopping and removing container ${name}"
              docker rm -f "${name}"
            fi
          done
          
          # 로그 디렉토리 권한 확인 및 생성
          sudo mkdir -p /var/logs/report_generator
          sudo chmod 755 /var/logs/report_generator
          
          # 새 컨테이너 실행
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
            "${IMAGE_NAME}:${IMAGE_TAG}"
        '''
      }
    }
        
    stage('Health Check') {
      steps {
        sh '''
          echo "Waiting for container to start..."
          sleep 20
          
          # 컨테이너 상태 확인
          docker ps | grep ${IMAGE_NAME} || (echo "Container not running" && exit 1)
          
          # Health check with retry (sh 호환 문법)
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
          
          # 로그 파일 생성 확인
          if [ -f "/var/logs/report_generator/report_generator.log" ]; then
            echo "📝 Log file created successfully"
          else
            echo "⚠️ Log file not found yet"
          fi
        '''
      }
    }
        
    stage('Integration Test') {
      steps {
        sh '''
          echo "Running integration tests..."
          
          # Team5-net 네트워크 연결 확인
          docker exec ${IMAGE_NAME} ping -c 2 elasticsearch || echo "Elasticsearch ping failed"
          
          # Prometheus 메트릭 확인
          if docker exec team5-prom wget -qO- http://${IMAGE_NAME}:8377/metrics | head -5; then
            echo "✅ Prometheus can scrape metrics"
          else
            echo "⚠️ Prometheus scraping issue"
          fi
        '''
      }
    }
        
    stage('Cleanup') {
      steps {
        script {
          sh "docker image prune -f"
        }
      }
    }
  }
  
  post {
    always {
      echo "Build #${env.BUILD_NUMBER} finished at ${new Date()}"
      // 로그 수집 상태 확인
      sh '''
        echo "=== Final Container Status ==="
        docker ps | grep ${IMAGE_NAME} || echo "Container not found"
        echo "=== Log Directory Status ==="
        ls -la /var/logs/report_generator/ || echo "Log directory not accessible"
      '''
    }
    success {
      echo "✅ Server3 Report Generator deployed successfully!"
      echo "🔗 Service URL: http://localhost:8377"
      echo "📊 Metrics URL: http://localhost:8377/metrics"
      echo "📝 Logs: /var/logs/report_generator/"
      echo "🔍 Kibana: Check report-generator-logs-* index"
    }
    failure {
      echo "❌ Deployment failed"
      sh '''
        echo "=== Container Logs ==="
        docker logs ${IMAGE_NAME} || echo "No container logs available"
        echo "=== System Status ==="
        docker ps -a | grep ${IMAGE_NAME}
        echo "=== Network Status ==="
        docker network inspect team5-net | grep ${IMAGE_NAME} || echo "Not in team5-net"
      '''
    }
  }
}