pipeline {
    agent any
    environment {
        DOCKERHUB_CREDENTIALS = credentials('docker-hub-credentials')
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
                    def app = docker.build("my-app:${env.BUILD_NUMBER}")
                }
            }
        }
        stage('Docker Login') {
            steps {
                script {
                    docker.withRegistry('', DOCKERHUB_CREDENTIALS) {
                    }
                }
            }
        }
        stage('Push Docker Image') {
            steps {
                script {
                    docker.image("my-app:${env.BUILD_NUMBER}").push()
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
