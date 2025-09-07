#!/usr/bin/env python3
"""Training script with spot instance interruption handling and automatic checkpointing."""

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
    def __init__(self, checkpoint_callback=None):
        self.checkpoint_callback = checkpoint_callback
        self.interrupted = False
        self.monitoring = False
        
    def start_monitoring(self):
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_interruption)
        monitor_thread.daemon = True
        monitor_thread.start()
        logger.info("Started spot instance interruption monitoring")
        
    def _monitor_interruption(self):
        while self.monitoring:
            try:
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
                    
                    time.sleep(30)
                    break
                    
            except requests.exceptions.RequestException:
                pass
            except Exception as e:
                logger.error(f"Error checking interruption status: {e}")
                
            time.sleep(5)
            
    def stop_monitoring(self):
        self.monitoring = False

class CheckpointManager:
    def __init__(self, s3_bucket, local_checkpoint_dir="/opt/checkpoints"):
        self.s3_bucket = s3_bucket
        self.local_dir = Path(local_checkpoint_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.s3_client = boto3.client('s3')
        
    def save_checkpoint(self, model_state, optimizer_state, metadata):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{timestamp}.pt"
        local_path = self.local_dir / checkpoint_name
        
        checkpoint_data = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'metadata': metadata,
            'timestamp': timestamp
        }
        
        with open(local_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved checkpoint: {checkpoint_name}")
        self.sync_to_s3(local_path)
        
    def sync_to_s3(self, local_path):
        try:
            s3_key = f"checkpoints/{local_path.name}"
            self.s3_client.upload_file(str(local_path), self.s3_bucket, s3_key)
            logger.info(f"Synced checkpoint to S3: s3://{self.s3_bucket}/{s3_key}")
        except Exception as e:
            logger.error(f"Failed to sync checkpoint to S3: {e}")
            
    def load_latest_checkpoint(self):
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix='checkpoints/'
            )
            
            if 'Contents' not in response:
                logger.info("No checkpoints found in S3")
                return None
                
            latest = max(response['Contents'], key=lambda x: x['LastModified'])
            s3_key = latest['Key']
            local_path = self.local_dir / Path(s3_key).name
            
            self.s3_client.download_file(self.s3_bucket, s3_key, str(local_path))
            logger.info(f"Downloaded checkpoint from S3: {s3_key}")
            
            return {'path': str(local_path)}
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from S3: {e}")
            return None

def mock_training_loop():
    s3_bucket = os.environ.get('S3_BUCKET')
    if not s3_bucket:
        logger.error("S3_BUCKET environment variable not set")
        return
        
    checkpoint_mgr = CheckpointManager(s3_bucket)
    latest_checkpoint = checkpoint_mgr.load_latest_checkpoint()
    start_step = 0
    
    if latest_checkpoint:
        logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
        start_step = 100
    
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
    
    try:
        for step in range(start_step, 1000):
            current_step = step
            
            if spot_handler.interrupted:
                logger.info("Training interrupted by spot instance termination")
                break
                
            time.sleep(1)
            
            if step % 10 == 0:
                logger.info(f"Training step {step}, loss: 0.5")
                
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