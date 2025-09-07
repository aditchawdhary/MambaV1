#!/bin/bash
# User data script for AWS EC2 training instances
# Sets up the training environment and handles spot instance interruptions

set -e

# Variables from Terraform
S3_BUCKET="${s3_bucket}"
PROJECT_NAME="${project_name}"
ENVIRONMENT="${environment}"

# Configuration
TRAINING_DIR="/opt/mamba-training"
CHECKPOINT_DIR="/opt/checkpoints"
LOG_FILE="/var/log/mamba-training-setup.log"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting Mamba training instance setup..."
log "Instance ID: $INSTANCE_ID"
log "S3 Bucket: $S3_BUCKET"

# Update system
log "Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Install required packages
log "Installing system dependencies..."
apt-get install -y \
    git \
    wget \
    curl \
    htop \
    nvtop \
    tmux \
    screen \
    awscli \
    jq \
    unzip \
    build-essential \
    python3-pip \
    python3-venv

# Install CloudWatch agent
log "Installing CloudWatch agent..."
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i amazon-cloudwatch-agent.deb

# Configure CloudWatch agent
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << EOF
{
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/mamba-training-setup.log",
                        "log_group_name": "/aws/ec2/${PROJECT_NAME}-${ENVIRONMENT}/training",
                        "log_stream_name": "{instance_id}/setup"
                    },
                    {
                        "file_path": "/opt/mamba-training/logs/training.log",
                        "log_group_name": "/aws/ec2/${PROJECT_NAME}-${ENVIRONMENT}/training",
                        "log_stream_name": "{instance_id}/training"
                    }
                ]
            }
        }
    },
    "metrics": {
        "namespace": "MambaTraining",
        "metrics_collected": {
            "cpu": {
                "measurement": ["cpu_usage_idle", "cpu_usage_iowait", "cpu_usage_user", "cpu_usage_system"],
                "metrics_collection_interval": 60
            },
            "disk": {
                "measurement": ["used_percent"],
                "metrics_collection_interval": 60,
                "resources": ["*"]
            },
            "mem": {
                "measurement": ["mem_used_percent"],
                "metrics_collection_interval": 60
            }
        }
    }
}
EOF

# Start CloudWatch agent
systemctl enable amazon-cloudwatch-agent
systemctl start amazon-cloudwatch-agent

# Create training directory
log "Setting up training directory..."
mkdir -p "$TRAINING_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$TRAINING_DIR/logs"
mkdir -p "$TRAINING_DIR/data"

# Clone the training repository (assuming it's in a git repo)
log "Setting up training code..."
cd "$TRAINING_DIR"

# Create a simple training script that handles checkpointing and spot interruptions
cat > "$TRAINING_DIR/train_with_spot_handling.py" << 'EOF'
#!/usr/bin/env python3
"""
Training script with spot instance interruption handling and automatic checkpointing.
"""

import os
import sys
import time
import signal
import json
import boto3
import requests
import threading
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/mamba-training/logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SpotInstanceHandler:
    """Handles spot instance interruption detection and graceful shutdown."""
    
    def __init__(self, checkpoint_callback=None):
        self.checkpoint_callback = checkpoint_callback
        self.interrupted = False
        self.monitoring = False
        
    def start_monitoring(self):
        """Start monitoring for spot instance interruption."""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_interruption)
        monitor_thread.daemon = True
        monitor_thread.start()
        logger.info("Started spot instance interruption monitoring")
        
    def _monitor_interruption(self):
        """Monitor AWS metadata for spot instance interruption notice."""
        while self.monitoring:
            try:
                # Check for spot instance interruption notice
                response = requests.get(
                    'http://169.254.169.254/latest/meta-data/spot/instance-action',
                    timeout=2
                )
                
                if response.status_code == 200:
                    logger.warning("Spot instance interruption detected!")
                    self.interrupted = True
                    
                    if self.checkpoint_callback:
                        logger.info("Triggering emergency checkpoint...")
                        self.checkpoint_callback()
                    
                    # Give some time for checkpoint to complete
                    time.sleep(30)
                    break
                    
            except requests.exceptions.RequestException:
                # No interruption notice (this is normal)
                pass
            except Exception as e:
                logger.error(f"Error checking interruption status: {e}")
                
            time.sleep(5)  # Check every 5 seconds
            
    def stop_monitoring(self):
        """Stop monitoring for interruptions."""
        self.monitoring = False

class CheckpointManager:
    """Manages model checkpoints with S3 sync."""
    
    def __init__(self, s3_bucket, local_checkpoint_dir="/opt/checkpoints"):
        self.s3_bucket = s3_bucket
        self.local_dir = Path(local_checkpoint_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.s3_client = boto3.client('s3')
        
    def save_checkpoint(self, model_state, optimizer_state, metadata):
        """Save checkpoint locally and sync to S3."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{timestamp}.pt"
        local_path = self.local_dir / checkpoint_name
        
        # Save checkpoint data
        checkpoint_data = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'metadata': metadata,
            'timestamp': timestamp
        }
        
        # In a real implementation, you would use torch.save here
        # torch.save(checkpoint_data, local_path)
        
        # For now, just save metadata as JSON
        with open(local_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved checkpoint: {checkpoint_name}")
        
        # Sync to S3
        self.sync_to_s3(local_path)
        
    def sync_to_s3(self, local_path):
        """Sync checkpoint to S3."""
        try:
            s3_key = f"checkpoints/{local_path.name}"
            self.s3_client.upload_file(str(local_path), self.s3_bucket, s3_key)
            logger.info(f"Synced checkpoint to S3: s3://{self.s3_bucket}/{s3_key}")
        except Exception as e:
            logger.error(f"Failed to sync checkpoint to S3: {e}")
            
    def load_latest_checkpoint(self):
        """Load the latest checkpoint from S3."""
        try:
            # List checkpoints in S3
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix='checkpoints/'
            )
            
            if 'Contents' not in response:
                logger.info("No checkpoints found in S3")
                return None
                
            # Find latest checkpoint
            latest = max(response['Contents'], key=lambda x: x['LastModified'])
            s3_key = latest['Key']
            local_path = self.local_dir / Path(s3_key).name
            
            # Download from S3
            self.s3_client.download_file(self.s3_bucket, s3_key, str(local_path))
            logger.info(f"Downloaded checkpoint from S3: {s3_key}")
            
            # Load checkpoint data
            # In a real implementation: return torch.load(local_path)
            return {'path': str(local_path)}
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from S3: {e}")
            return None

def mock_training_loop():
    """Mock training loop for demonstration."""
    s3_bucket = os.environ.get('S3_BUCKET')
    if not s3_bucket:
        logger.error("S3_BUCKET environment variable not set")
        return
        
    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(s3_bucket)
    
    # Load latest checkpoint if available
    latest_checkpoint = checkpoint_mgr.load_latest_checkpoint()
    start_step = 0
    
    if latest_checkpoint:
        logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
        # In real implementation, load model and optimizer state
        start_step = 100  # Mock resume step
    
    # Setup spot instance handler
    def emergency_checkpoint():
        logger.info("Performing emergency checkpoint due to spot interruption...")
        checkpoint_mgr.save_checkpoint(
            model_state={'mock': 'model_state'},
            optimizer_state={'mock': 'optimizer_state'},
            metadata={
                'step': current_step,
                'loss': 0.5,
                'emergency': True
            }
        )
    
    spot_handler = SpotInstanceHandler(checkpoint_callback=emergency_checkpoint)
    spot_handler.start_monitoring()
    
    # Mock training loop
    try:
        for step in range(start_step, 1000):
            current_step = step
            
            # Check for interruption
            if spot_handler.interrupted:
                logger.info("Training interrupted by spot instance termination")
                break
                
            # Mock training step
            time.sleep(1)  # Simulate training time
            
            if step % 10 == 0:
                logger.info(f"Training step {step}, loss: 0.5")
                
            # Regular checkpoint every 50 steps
            if step % 50 == 0:
                checkpoint_mgr.save_checkpoint(
                    model_state={'mock': 'model_state'},
                    optimizer_state={'mock': 'optimizer_state'},
                    metadata={
                        'step': step,
                        'loss': 0.5,
                        'emergency': False
                    }
                )
                
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        spot_handler.stop_monitoring()
        logger.info("Training completed")

if __name__ == "__main__":
    logger.info("Starting Mamba training with spot instance handling...")
    mock_training_loop()
EOF

# Make training script executable
chmod +x "$TRAINING_DIR/train_with_spot_handling.py"

# Create systemd service for automatic training restart
log "Creating training service..."
cat > /etc/systemd/system/mamba-training.service << EOF
[Unit]
Description=Mamba Training Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$TRAINING_DIR
Environment=S3_BUCKET=$S3_BUCKET
Environment=PYTHONPATH=$TRAINING_DIR
ExecStart=/usr/bin/python3 $TRAINING_DIR/train_with_spot_handling.py
Restart=always
RestartSec=10
StandardOutput=append:$TRAINING_DIR/logs/training.log
StandardError=append:$TRAINING_DIR/logs/training.log

[Install]
WantedBy=multi-user.target
EOF

# Create spot instance termination handler
log "Setting up spot instance termination handler..."
cat > "$TRAINING_DIR/spot_termination_handler.sh" << 'EOF'
#!/bin/bash
# Spot instance termination handler

LOG_FILE="/var/log/spot-termination.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check for spot instance termination notice
while true; do
    if curl -s http://169.254.169.254/latest/meta-data/spot/instance-action | grep -q terminate; then
        log "Spot instance termination notice received"
        
        # Stop training service gracefully
        log "Stopping training service..."
        systemctl stop mamba-training
        
        # Wait a bit for graceful shutdown
        sleep 30
        
        log "Instance ready for termination"
        break
    fi
    
    sleep 5
done
EOF

chmod +x "$TRAINING_DIR/spot_termination_handler.sh"

# Create systemd service for spot termination handler
cat > /etc/systemd/system/spot-termination-handler.service << EOF
[Unit]
Description=Spot Instance Termination Handler
After=network.target

[Service]
Type=simple
User=root
ExecStart=$TRAINING_DIR/spot_termination_handler.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Install Python dependencies
log "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install boto3 requests torch numpy

# Set up environment variables
log "Setting up environment..."
cat >> /etc/environment << EOF
S3_BUCKET=$S3_BUCKET
PROJECT_NAME=$PROJECT_NAME
ENVIRONMENT=$ENVIRONMENT
TRAINING_DIR=$TRAINING_DIR
CHECKPOINT_DIR=$CHECKPOINT_DIR
EOF

# Enable and start services
log "Starting services..."
systemctl daemon-reload
systemctl enable mamba-training
systemctl enable spot-termination-handler
systemctl start spot-termination-handler

# Create startup script that runs on boot
cat > "$TRAINING_DIR/startup.sh" << 'EOF'
#!/bin/bash
# Startup script that runs on instance boot

LOG_FILE="/var/log/mamba-training-startup.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Instance startup - checking for training resumption..."

# Wait for network to be ready
sleep 30

# Check if we should start training automatically
if [ "$ENVIRONMENT" = "prod" ] || [ -f "/opt/mamba-training/auto_start" ]; then
    log "Starting training service automatically..."
    systemctl start mamba-training
else
    log "Training service not started automatically (dev environment)"
fi

log "Startup complete"
EOF

chmod +x "$TRAINING_DIR/startup.sh"

# Add startup script to crontab
echo "@reboot $TRAINING_DIR/startup.sh" | crontab -

# Create helper scripts
log "Creating helper scripts..."

# Script to manually start training
cat > "$TRAINING_DIR/start_training.sh" << 'EOF'
#!/bin/bash
echo "Starting Mamba training..."
systemctl start mamba-training
systemctl status mamba-training
EOF

# Script to stop training
cat > "$TRAINING_DIR/stop_training.sh" << 'EOF'
#!/bin/bash
echo "Stopping Mamba training..."
systemctl stop mamba-training
systemctl status mamba-training
EOF

# Script to check training status
cat > "$TRAINING_DIR/check_status.sh" << 'EOF'
#!/bin/bash
echo "=== Training Service Status ==="
systemctl status mamba-training

echo -e "\n=== Recent Training Logs ==="
tail -n 20 /opt/mamba-training/logs/training.log

echo -e "\n=== GPU Status ==="
nvidia-smi

echo -e "\n=== Disk Usage ==="
df -h

echo -e "\n=== S3 Sync Status ==="
aws s3 ls s3://$S3_BUCKET/checkpoints/ --human-readable
EOF

# Make helper scripts executable
chmod +x "$TRAINING_DIR"/*.sh

# Set up log rotation
log "Setting up log rotation..."
cat > /etc/logrotate.d/mamba-training << EOF
$TRAINING_DIR/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF

# Final setup
log "Final setup steps..."
chown -R root:root "$TRAINING_DIR"
chown -R root:root "$CHECKPOINT_DIR"

# Create a flag file to indicate setup is complete
touch "$TRAINING_DIR/setup_complete"

log "Mamba training instance setup completed successfully!"
log "Training directory: $TRAINING_DIR"
log "Checkpoint directory: $CHECKPOINT_DIR"
log "S3 bucket: $S3_BUCKET"
log ""
log "To start training manually: $TRAINING_DIR/start_training.sh"
log "To check status: $TRAINING_DIR/check_status.sh"
log "To stop training: $TRAINING_DIR/stop_training.sh"

# Send completion notification to CloudWatch
aws cloudwatch put-metric-data \
    --namespace "MambaTraining" \
    --metric-data MetricName=InstanceSetupComplete,Value=1,Unit=Count \
    --region "$REGION" || true

log "Instance setup complete and ready for training!"
EOF