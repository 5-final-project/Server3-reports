pipeline {
  agent { label 'team5' }
  
  environment {
    IMAGE_NAME = "server3-report"
    IMAGE_TAG = "${env.BUILD_NUMBER}"
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
          # Ensure legacy container names are removed
          legacy_names=("great_villani" "server3-report-generator")
          for name in "${legacy_names[@]}"; do
            if docker ps -a --filter "name=^/${name}$" --format "{{.Names}}" | grep -q "^${name}$"; then
              echo "Stopping and removing legacy container ${name}"
              docker rm -f "${name}"
            fi
          done
          
          # Check if a container with this name exists (exact match)
          existing=$(docker ps -a --filter "name=^/${IMAGE_NAME}$" --format "{{.Names}}")
          if [ "$existing" = "${IMAGE_NAME}" ]; then
            echo "Stopping and removing existing container ${IMAGE_NAME}"
            docker rm -f "${IMAGE_NAME}"
          fi
          
          # Run new container (CPU-based, no GPU needed)
          docker run -d \
            --name "${IMAGE_NAME}" \
            --network team5-net \
            -p 8377:8377 \
            -v /var/logs/report_generator:/var/logs/report_generator \
            -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
            -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
            -e BUCKET_NAME=${BUCKET_NAME} \
            -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
            "${IMAGE_NAME}:${IMAGE_TAG}"
        '''
      }
    }
        
    stage('Health Check') {
      steps {
        sh '''
          echo "Waiting for container to start..."
          sleep 10
          
          # Health check
          if curl -f http://localhost:8377/ > /dev/null 2>&1; then
            echo "✅ Server3 Report Generator is running successfully"
          else
            echo "❌ Health check failed"
            docker logs ${IMAGE_NAME}
            exit 1
          fi
        '''
      }
    }
        
    stage('Cleanup') {
      steps {
        script {
          // Remove dangling images
          sh "docker image prune -f"
        }
      }
    }
  }
  
  post {
    always {
      echo "Build #${env.BUILD_NUMBER} finished at ${new Date()}"
    }
    success {
      echo "✅ Server3 Report Generator deployed successfully!"
      echo "🔗 Service URL: http://localhost:8377"
      echo "📊 Metrics URL: http://localhost:8377/metrics"
    }
    failure {
      echo "❌ Deployment failed"
      sh "docker logs ${IMAGE_NAME} || echo 'No container logs available'"
    }
  }
}