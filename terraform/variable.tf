variable "gcp_project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP Region for Cloud Run"
  type        = string
  default     = "us-central1"
}

variable "dockerhub_username" {
  description = "Docker Hub Username"
  type        = string
}

variable "dockerhub_token" {
  description = "Docker Hub Access Token"
  type        = string
}
