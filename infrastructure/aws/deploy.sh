#!/bin/bash
# Deployment script for AWS Mamba training infrastructure

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if terraform is installed
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is not installed. Please install Terraform first."
        log_info "Visit: https://www.terraform.io/downloads.html"
        exit 1
    fi
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install AWS CLI first."
        log_info "Visit: https://aws.amazon.com/cli/"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    # Check if terraform.tfvars exists
    if [ ! -f "$SCRIPT_DIR/terraform.tfvars" ]; then
        log_error "terraform.tfvars not found!"
        log_info "Please copy terraform.tfvars.example to terraform.tfvars and customize it:"
        log_info "  cp terraform.tfvars.example terraform.tfvars"
        log_info "  # Edit terraform.tfvars with your settings"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Initialize Terraform
init_terraform() {
    log_info "Initializing Terraform..."
    cd "$SCRIPT_DIR"
    
    terraform init
    
    log_success "Terraform initialized"
}

# Plan deployment
plan_deployment() {
    log_info "Planning Terraform deployment..."
    cd "$SCRIPT_DIR"
    
    terraform plan -out=tfplan
    
    log_success "Terraform plan created"
}

# Apply deployment
apply_deployment() {
    log_info "Applying Terraform deployment..."
    cd "$SCRIPT_DIR"
    
    if [ -f "tfplan" ]; then
        terraform apply tfplan
        rm -f tfplan
    else
        log_warning "No plan file found, running apply directly..."
        terraform apply -auto-approve
    fi
    
    log_success "Infrastructure deployed successfully!"
}

# Show outputs
show_outputs() {
    log_info "Deployment outputs:"
    cd "$SCRIPT_DIR"
    
    echo ""
    terraform output
    echo ""
    
    # Get S3 bucket name for convenience
    S3_BUCKET=$(terraform output -raw s3_bucket_name 2>/dev/null || echo "")
    if [ -n "$S3_BUCKET" ]; then
        log_info "S3 Bucket for checkpoints: $S3_BUCKET"
        log_info "Upload your training code and data:"
        log_info "  aws s3 cp /path/to/your/code s3://$S3_BUCKET/code/ --recursive"
        log_info "  aws s3 cp /path/to/your/data s3://$S3_BUCKET/data/ --recursive"
    fi
    
    echo ""
    log_info "To connect to training instances:"
    log_info "  aws ec2 describe-instances --filters 'Name=tag:Project,Values=mamba-training' --query 'Reservations[*].Instances[*].[InstanceId,PublicIpAddress,State.Name]' --output table"
    log_info "  ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>"
    
    echo ""
    log_info "To monitor training:"
    log_info "  aws logs tail /aws/ec2/mamba-training-dev/training --follow"
    
    echo ""
    log_info "To scale instances:"
    log_info "  aws autoscaling set-desired-capacity --auto-scaling-group-name mamba-training-dev-training-asg --desired-capacity 2"
}

# Destroy infrastructure
destroy_infrastructure() {
    log_warning "This will destroy all infrastructure resources!"
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        log_info "Destroying infrastructure..."
        cd "$SCRIPT_DIR"
        terraform destroy -auto-approve
        log_success "Infrastructure destroyed"
    else
        log_info "Destruction cancelled"
    fi
}

# Validate configuration
validate_config() {
    log_info "Validating Terraform configuration..."
    cd "$SCRIPT_DIR"
    
    terraform validate
    
    log_success "Configuration is valid"
}

# Show help
show_help() {
    echo "AWS Mamba Training Infrastructure Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy     - Deploy the complete infrastructure (init + plan + apply)"
    echo "  init       - Initialize Terraform"
    echo "  plan       - Create deployment plan"
    echo "  apply      - Apply deployment plan"
    echo "  outputs    - Show deployment outputs"
    echo "  validate   - Validate Terraform configuration"
    echo "  destroy    - Destroy all infrastructure"
    echo "  help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 deploy              # Full deployment"
    echo "  $0 plan                # Just create a plan"
    echo "  $0 outputs             # Show current outputs"
    echo "  $0 destroy             # Destroy everything"
    echo ""
    echo "Prerequisites:"
    echo "  - Terraform installed"
    echo "  - AWS CLI installed and configured"
    echo "  - terraform.tfvars file configured"
}

# Main script logic
main() {
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            init_terraform
            plan_deployment
            apply_deployment
            show_outputs
            ;;
        "init")
            check_prerequisites
            init_terraform
            ;;
        "plan")
            check_prerequisites
            plan_deployment
            ;;
        "apply")
            check_prerequisites
            apply_deployment
            show_outputs
            ;;
        "outputs")
            show_outputs
            ;;
        "validate")
            validate_config
            ;;
        "destroy")
            check_prerequisites
            destroy_infrastructure
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"