# Terraform configuration for AWS GPU training infrastructure
# Supports spot instances, auto-scaling, and checkpoint storage

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Name of the project (used for resource naming)"
  type        = string
  default     = "mamba-training"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "instance_types" {
  description = "List of GPU instance types to use (in order of preference)"
  type        = list(string)
  default     = ["g4dn.xlarge", "g4dn.2xlarge", "g4dn.4xlarge", "p3.2xlarge"]
}

variable "max_spot_price" {
  description = "Maximum price to pay for spot instances (USD per hour)"
  type        = string
  default     = "0.50"
}

variable "min_instances" {
  description = "Minimum number of instances in auto scaling group"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of instances in auto scaling group"
  type        = number
  default     = 4
}

variable "desired_instances" {
  description = "Desired number of instances in auto scaling group"
  type        = number
  default     = 1
}

variable "key_pair_name" {
  description = "Name of AWS key pair for SSH access"
  type        = string
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to SSH to instances"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict this in production!
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]
  
  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch *"]
  }
  
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# VPC and Networking
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-vpc"
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-igw"
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_subnet" "public" {
  count = min(length(data.aws_availability_zones.available.names), 3)
  
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-public-${count.index + 1}"
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-public-rt"
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)
  
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# Security Groups
resource "aws_security_group" "training_instances" {
  name_prefix = "${var.project_name}-${var.environment}-training-"
  vpc_id      = aws_vpc.main.id
  
  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }
  
  # TensorBoard (optional)
  ingress {
    from_port   = 6006
    to_port     = 6006
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }
  
  # Jupyter (optional)
  ingress {
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }
  
  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-training-sg"
    Project     = var.project_name
    Environment = var.environment
  }
}

# S3 Bucket for checkpoints and datasets
resource "aws_s3_bucket" "training_data" {
  bucket = "${var.project_name}-${var.environment}-training-data-${random_id.bucket_suffix.hex}"
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-training-data"
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "training_data" {
  bucket = aws_s3_bucket.training_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "training_data" {
  bucket = aws_s3_bucket.training_data.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# IAM Role for EC2 instances
resource "aws_iam_role" "training_instance_role" {
  name = "${var.project_name}-${var.environment}-training-instance-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-training-instance-role"
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_iam_role_policy" "training_instance_policy" {
  name = "${var.project_name}-${var.environment}-training-instance-policy"
  role = aws_iam_role.training_instance_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.training_data.arn,
          "${aws_s3_bucket.training_data.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:DescribeSpotInstanceRequests",
          "ec2:DescribeSpotPriceHistory"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "training_instance_profile" {
  name = "${var.project_name}-${var.environment}-training-instance-profile"
  role = aws_iam_role.training_instance_role.name
}

# Launch Template for training instances
resource "aws_launch_template" "training_template" {
  name_prefix   = "${var.project_name}-${var.environment}-training-"
  image_id      = data.aws_ami.deep_learning.id
  key_name      = var.key_pair_name
  
  vpc_security_group_ids = [aws_security_group.training_instances.id]
  
  iam_instance_profile {
    name = aws_iam_instance_profile.training_instance_profile.name
  }
  
  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    s3_bucket = aws_s3_bucket.training_data.bucket
    project_name = var.project_name
    environment = var.environment
  }))
  
  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "${var.project_name}-${var.environment}-training-instance"
      Project     = var.project_name
      Environment = var.environment
    }
  }
  
  # Enable detailed monitoring
  monitoring {
    enabled = true
  }
  
  # Use spot instances
  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price = var.max_spot_price
      spot_instance_type = "one-time"
    }
  }
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-training-template"
    Project     = var.project_name
    Environment = var.environment
  }
}

# Auto Scaling Group for spot instances
resource "aws_autoscaling_group" "training_asg" {
  name                = "${var.project_name}-${var.environment}-training-asg"
  vpc_zone_identifier = aws_subnet.public[*].id
  min_size            = var.min_instances
  max_size            = var.max_instances
  desired_capacity    = var.desired_instances
  
  mixed_instances_policy {
    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.training_template.id
        version           = "$Latest"
      }
      
      dynamic "override" {
        for_each = var.instance_types
        content {
          instance_type = override.value
        }
      }
    }
    
    instances_distribution {
      on_demand_base_capacity                  = 0
      on_demand_percentage_above_base_capacity = 0
      spot_allocation_strategy                 = "diversified"
    }
  }
  
  tag {
    key                 = "Name"
    value               = "${var.project_name}-${var.environment}-training-asg"
    propagate_at_launch = false
  }
  
  tag {
    key                 = "Project"
    value               = var.project_name
    propagate_at_launch = true
  }
  
  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }
}

# CloudWatch Log Group for training logs
resource "aws_cloudwatch_log_group" "training_logs" {
  name              = "/aws/ec2/${var.project_name}-${var.environment}/training"
  retention_in_days = 14
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-training-logs"
    Project     = var.project_name
    Environment = var.environment
  }
}

# Outputs
output "s3_bucket_name" {
  description = "Name of the S3 bucket for training data and checkpoints"
  value       = aws_s3_bucket.training_data.bucket
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "security_group_id" {
  description = "ID of the training instances security group"
  value       = aws_security_group.training_instances.id
}

output "launch_template_id" {
  description = "ID of the launch template"
  value       = aws_launch_template.training_template.id
}

output "autoscaling_group_name" {
  description = "Name of the auto scaling group"
  value       = aws_autoscaling_group.training_asg.name
}

output "cloudwatch_log_group" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.training_logs.name
}