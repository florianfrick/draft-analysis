output "ecr_repository_uri" {
  description = "Private ECR repo in the function Region. Push the slim image here."
  value       = aws_ecr_repository.web.repository_url
}

output "image_uri" {
  description = "Exactly what Lambda will deploy."
  value       = aws_lambda_function.web.image_uri
}

output "function_url" {
  description = "Direct iframe src: the function serves the static app and its API."
  value       = aws_lambda_function_url.web.function_url
}

output "embed_url" {
  description = "Alias of function_url, kept for the portfolio iframe."
  value       = aws_lambda_function_url.web.function_url
}
