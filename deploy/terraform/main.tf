data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "web" {
  name               = "${var.name}-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

# No VPC, no NAT Gateway, no DynamoDB: the only extra permission is log
# writing. This is what keeps the deployment inside the Always Free limits.
resource "aws_iam_role_policy_attachment" "logs" {
  role       = aws_iam_role.web.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_cloudwatch_log_group" "web" {
  name              = "/aws/lambda/${var.name}"
  retention_in_days = 7
}

# Lambda requires the function image to live in an ECR repository in the
# SAME Region as the function. Public ECR is always-free but is a global
# us-east-1 registry, so it cannot be the deployment source for a Lambda in
# us-west-2. A private repo is used here; the lifecycle rule keeps only the
# two most recent images (~3.1 GB each) to bound storage.
resource "aws_ecr_repository" "web" {
  name                 = var.name
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

resource "aws_ecr_lifecycle_policy" "web" {
  repository = aws_ecr_repository.web.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the current image"
      selection = {
        # "any" also covers the old image left untagged by a mutable "latest"
        # re-push. ECR now requires a tag pattern whenever tagStatus is "tagged".
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 1
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_lambda_function" "web" {
  function_name = var.name
  role          = aws_iam_role.web.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.web.repository_url}:${var.image_tag}"
  architectures = ["x86_64"]
  memory_size   = var.memory_size
  timeout       = var.timeout

  environment {
    variables = {
      WEB_ORIGIN      = var.web_origin
      LLM_BASE_URL    = var.worker_base_url
      LLM_MODEL       = var.llm_model
      LLM_API_KEY     = var.llm_token
      LLM_TIMEOUT     = "45"
      AGENT_CALLS     = tostring(var.agent_calls)
      OMP_NUM_THREADS = "2"
    }
  }

  depends_on = [aws_cloudwatch_log_group.web,
  aws_iam_role_policy_attachment.logs]
}

# Function URLs have no API Gateway charge. Auth is NONE because an iframe
# cannot sign IAM requests. LLM spend is bounded by the Worker's turn cap;
# raw request volume is bounded only by Lambda's permanent free tier.
resource "aws_lambda_function_url" "web" {
  function_name      = aws_lambda_function.web.function_name
  authorization_type = "NONE"

  cors {
    allow_credentials = false
    allow_origins     = [var.web_origin]
    # Lambda answers OPTIONS preflight itself; AWS rejects OPTIONS here.
    allow_methods  = ["GET", "POST"]
    allow_headers  = ["content-type"]
    expose_headers = []
    max_age        = 600
  }
}
