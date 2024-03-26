pipeline {
    agent any
    environment {
        // DOCKERHUB_CREDENTIALS is now just an ID, not the actual credentials.
        DOCKERHUB_CREDENTIALS = 'docker-hub-credentials' // ID of your credentials in Jenkins
        RECIPIENT = 'i200895@nu.edu.pk'
    }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    // You can add 'docker' to the registry URL if you have namespaced your image
                    def app = docker.build("zaynshayq/my-app:${env.BUILD_NUMBER}")
                }
            }
        }
        stage('Docker Login') {
            steps {
                script {
                    docker.withRegistry('https://registry.hub.docker.com', DOCKERHUB_CREDENTIALS) {
                        // Docker login is handled automatically inside this block
                    }
                }
            }
        }
        stage('Push Docker Image') {
            steps {
                script {
                    docker.withRegistry('https://registry.hub.docker.com', DOCKERHUB_CREDENTIALS) {
                        docker.image("zaynshayq/my-app:${env.BUILD_NUMBER}").push()
                    }
                }
            }
        }
    }
    post {
        success {
            emailext(
                to: "${RECIPIENT}",
                subject: "SUCCESSFUL: Docker Image Build #${env.BUILD_NUMBER}",
                body: "The Docker image build was successful.\n\nBuild Number: ${env.BUILD_NUMBER}\nJob Name: ${env.JOB_NAME}"
            )
        }
        failure {
            emailext(
                to: "${RECIPIENT}",
                subject: "FAILED: Docker Image Build #${env.BUILD_NUMBER}",
                body: "There was a failure building the Docker image.\n\nBuild Number: ${env.BUILD_NUMBER}\nJob Name: ${env.JOB_NAME}"
            )
        }
    }
}

