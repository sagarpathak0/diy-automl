import os
import time
import shutil
import argparse
from datetime import datetime, timedelta

def cleanup_old_sessions(uploads_dir="uploads", max_age_hours=24):
    """Remove session folders older than max_age_hours"""
    now = time.time()
    count = 0
    
    if not os.path.exists(uploads_dir):
        print(f"Directory {uploads_dir} does not exist. Nothing to clean.")
        return
    
    for session_dir in os.listdir(uploads_dir):
        session_path = os.path.join(uploads_dir, session_dir)
        if os.path.isdir(session_path) and session_dir.startswith("session_"):
            # Check folder age
            creation_time = os.path.getctime(session_path)
            age_hours = (now - creation_time) / 3600
            
            if age_hours > max_age_hours:
                print(f"Removing {session_dir} (Age: {age_hours:.1f} hours)")
                shutil.rmtree(session_path)
                count += 1
    
    print(f"Cleaned up {count} session folders older than {max_age_hours} hours")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up old session folders")
    parser.add_argument("--hours", type=int, default=24, 
                        help="Maximum age in hours (default: 24)")
    parser.add_argument("--dir", type=str, default="uploads",
                        help="Path to uploads directory (default: uploads)")
    
    args = parser.parse_args()
    cleanup_old_sessions(args.dir, args.hours)