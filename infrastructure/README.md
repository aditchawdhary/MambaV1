# Mamba Training Infrastructure

This directory contains Infrastructure as Code (IaC) for deploying GPU training infrastructure on AWS with spot instance support, automatic checkpointing, and cost optimization.

## Features

- **Spot Instance Management**: Automatic spot instance provisioning with interruption handling
- **Auto-scaling**: Scale from 0 to N instances based on training demand
- **Checkpoint Resilience**: Automatic S3 sync for model checkpoints
- **Cost Optimization**: Up to 90% cost savings using spot instances
- **Multi-GPU Support**: Distributed training across multiple instances
- **Monitoring**: CloudWatch integration for logs and metrics

## Quick Start

### 1. Prerequisites

```bash
# Install Terraform
brew install terraform  # macOS
# or download from https://www.terraform.io/downloads.html

# Install AWS CLI
brew install awscli     # macOS
# or download from https://aws.amazon.com/cli/

# Configure AWS credentials
aws configure
```

### 2. Setup Configuration

```bash
# Copy example configuration
cd infrastructure/aws
cp terraform.tfvars.example terraform.tfvars

# Edit configuration (IMPORTANT: Set your key pair name!)
vim terraform.tfvars
```

**Key settings to customize:**
- `key_pair_name`: Your AWS key pair for SSH access
- `allowed_ssh_cidrs`: Restrict SSH access (don't use 0.0.0.0/0 in production!)
- `max_spot_price`: Maximum price you're willing to pay per hour
- `instance_types`: GPU instance types in order of preference

### 3. Deploy Infrastructure

```bash
# Deploy everything
./deploy.sh deploy

# Or step by step:
./deploy.sh init      # Initialize Terraform
./deploy.sh plan      # Create deployment plan
./deploy.sh apply     # Apply changes
./deploy.sh outputs   # Show deployment info
```

### 4. Start Training

```bash
# Check running instances
aws ec2 describe-instances \
  --filters 'Name=tag:Project,Values=mamba-training' \
  --query 'Reservations[*].Instances[*].[InstanceId,PublicIpAddress,State.Name]' \
  --output table

# SSH to instance
ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>

# On the instance, check status
/opt/mamba-training/check_status.sh

# Start training manually (if not auto-started)
/opt/mamba-training/start_training.sh
```

## Architecture

### Infrastructure Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Auto Scaling  │    │  Spot Instances  │    │   S3 Bucket     │
│     Group       │───▶│  (GPU Training)  │───▶│  (Checkpoints)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Launch Template│    │   CloudWatch     │    │   IAM Roles     │
│  (Spot Config)  │    │   (Monitoring)   │    │ (Permissions)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Cost Optimization Strategy

1. **Spot Instances**: 50-90% cost savings vs on-demand
2. **Auto-scaling**: Scale to zero when not training
3. **Mixed Instance Types**: Increase availability across instance types
4. **Efficient Checkpointing**: Resume training quickly after interruptions

### Spot Instance Handling

The infrastructure includes sophisticated spot instance interruption handling:

1. **Interruption Detection**: Monitors AWS metadata for termination notices
2. **Emergency Checkpointing**: Automatically saves model state before termination
3. **Graceful Shutdown**: Allows 30 seconds for checkpoint completion
4. **Auto-restart**: New instances automatically resume from latest checkpoint

## Instance Types and Pricing

| Instance Type | GPU | vCPU | RAM | On-Demand | Typical Spot |
|---------------|-----|------|-----|-----------|--------------|
| g4dn.xlarge   | 1x T4 | 4 | 16GB | $0.526/hr | $0.15-0.25/hr |
| g4dn.2xlarge  | 1x T4 | 8 | 32GB | $0.752/hr | $0.20-0.35/hr |
| g4dn.4xlarge  | 1x T4 | 16 | 64GB | $1.204/hr | $0.30-0.50/hr |
| p3.2xlarge    | 1x V100 | 8 | 61GB | $3.06/hr | $0.90-1.50/hr |

*Spot prices vary by region and demand*

## Configuration Options

### Environment-Specific Configs

**Development Environment:**
```hcl
instance_types = ["g4dn.xlarge"]
max_spot_price = "0.30"
max_instances = 1
desired_instances = 0  # Start with no instances
```

**Production Environment:**
```hcl
instance_types = ["g4dn.2xlarge", "g4dn.4xlarge", "p3.2xlarge"]
max_spot_price = "1.00"
max_instances = 8
desired_instances = 2  # Always have 2 instances running
```

### Security Configuration

**Restrict SSH Access (Recommended):**
```hcl
allowed_ssh_cidrs = [
  "203.0.113.0/24",    # Your office network
  "198.51.100.0/24"    # Your home network
]
```

## Monitoring and Logging

### CloudWatch Integration

- **Logs**: Training logs automatically sent to CloudWatch
- **Metrics**: CPU, memory, disk usage tracked
- **Custom Metrics**: Training progress and model metrics

### View Logs

```bash
# Follow training logs in real-time
aws logs tail /aws/ec2/mamba-training-dev/training --follow

# View setup logs
aws logs tail /aws/ec2/mamba-training-dev/training --filter-pattern "setup"
```

### Monitor Costs

```bash
# Check current spot prices
aws ec2 describe-spot-price-history \
  --instance-types g4dn.xlarge g4dn.2xlarge \
  --product-descriptions "Linux/UNIX" \
  --max-items 10

# Monitor spending
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost
```

## Checkpoint Management

### S3 Checkpoint Storage

Checkpoints are automatically synced to S3 for durability:

```
s3://your-bucket/
├── checkpoints/
│   ├── checkpoint_20240101_120000.pt
│   ├── checkpoint_20240101_130000.pt
│   └── latest_checkpoint.pt
├── data/
│   └── training_datasets/
└── logs/
    └── training_logs/
```

### Manual Checkpoint Operations

```bash
# List checkpoints
aws s3 ls s3://your-bucket/checkpoints/ --human-readable

# Download specific checkpoint
aws s3 cp s3://your-bucket/checkpoints/checkpoint_20240101_120000.pt ./

# Upload checkpoint
aws s3 cp ./my_checkpoint.pt s3://your-bucket/checkpoints/
```

## Scaling Operations

### Manual Scaling

```bash
# Scale up to 4 instances
aws autoscaling set-desired-capacity \
  --auto-scaling-group-name mamba-training-dev-training-asg \
  --desired-capacity 4

# Scale down to 0 (stop all training)
aws autoscaling set-desired-capacity \
  --auto-scaling-group-name mamba-training-dev-training-asg \
  --desired-capacity 0
```

### Automatic Scaling

You can configure automatic scaling based on:
- Training queue depth
- CloudWatch metrics
- Time-based schedules

## Troubleshooting

### Common Issues

**1. Spot Instance Interruptions**
```bash
# Check interruption history
aws ec2 describe-spot-instance-requests \
  --filters 'Name=state,Values=cancelled'
```

**2. Training Not Starting**
```bash
# SSH to instance and check logs
ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>
sudo journalctl -u mamba-training -f
```

**3. High Costs**
```bash
# Check if instances are running when not needed
aws ec2 describe-instances \
  --filters 'Name=instance-state-name,Values=running' \
  --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,LaunchTime]'
```

### Emergency Procedures

**Stop All Training (Emergency):**
```bash
# Scale down to 0 instances immediately
aws autoscaling set-desired-capacity \
  --auto-scaling-group-name mamba-training-dev-training-asg \
  --desired-capacity 0

# Terminate all running instances
aws ec2 terminate-instances \
  --instance-ids $(aws ec2 describe-instances \
    --filters 'Name=tag:Project,Values=mamba-training' 'Name=instance-state-name,Values=running' \
    --query 'Reservations[*].Instances[*].InstanceId' --output text)
```

## Cleanup

### Destroy Infrastructure

```bash
# Destroy all resources (WARNING: This deletes everything!)
./deploy.sh destroy
```

### Partial Cleanup

```bash
# Just scale down instances (keep infrastructure)
aws autoscaling set-desired-capacity \
  --auto-scaling-group-name mamba-training-dev-training-asg \
  --desired-capacity 0
```

## Advanced Usage

### Multi-Region Deployment

Deploy in multiple regions for better spot availability:

```bash
# Deploy in us-west-2
cd infrastructure/aws
terraform workspace new us-west-2
terraform apply -var="aws_region=us-west-2"

# Deploy in us-east-1
terraform workspace new us-east-1
terraform apply -var="aws_region=us-east-1"
```

### Custom AMI

Create a custom AMI with your training code pre-installed:

```bash
# Launch instance, install your code, then create AMI
aws ec2 create-image \
  --instance-id i-1234567890abcdef0 \
  --name "mamba-training-custom-$(date +%Y%m%d)" \
  --description "Custom AMI with Mamba training code"
```

### Integration with CI/CD

Integrate with GitHub Actions or other CI/CD systems:

```yaml
# .github/workflows/deploy-training.yml
name: Deploy Training Infrastructure
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v1
      - name: Deploy Infrastructure
        run: |
          cd infrastructure/aws
          ./deploy.sh deploy
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review AWS CloudWatch logs
3. Check Terraform state and outputs
4. Consult AWS documentation for spot instances

## Security Best Practices

1. **Restrict SSH Access**: Never use 0.0.0.0/0 in production
2. **Use IAM Roles**: Avoid hardcoded AWS credentials
3. **Enable VPC Flow Logs**: Monitor network traffic
4. **Regular Updates**: Keep AMIs and packages updated
5. **Encrypt Storage**: Enable EBS encryption for sensitive data