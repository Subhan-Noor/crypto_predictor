#!/bin/bash

# Crypto Price Prediction - Cron Automation Script
#
# This script is designed to be run via cron for daily automation.
# It handles data ingestion and prediction generation with error handling
# and logging suitable for unattended execution.
#
# Cron Example (daily at 6:00 AM):
# 0 6 * * * /path/to/capstone/backend/scripts/cron_automation.sh full
#
# Usage:
#   ./cron_automation.sh full           # Complete pipeline
#   ./cron_automation.sh data           # Data ingestion only  
#   ./cron_automation.sh predictions    # Predictions only
#   ./cron_automation.sh health         # Health check only

set -euo pipefail  # Exit on error, undefined vars, and pipe failures

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$BACKEND_DIR")"
LOG_DIR="$BACKEND_DIR/logs"
PYTHON_ENV="$BACKEND_DIR/venv"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Log files
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/automation_${TIMESTAMP}.log"
ERROR_LOG="$LOG_DIR/automation_errors_${TIMESTAMP}.log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE" "$ERROR_LOG"
}

# Function to check if virtual environment exists
check_venv() {
    if [ ! -d "$PYTHON_ENV" ]; then
        error_log "Python virtual environment not found at $PYTHON_ENV"
        error_log "Please run: cd $BACKEND_DIR && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
}

# Function to activate virtual environment
activate_venv() {
    log "Activating Python virtual environment..."
    # shellcheck source=/dev/null
    source "$PYTHON_ENV/bin/activate"
    
    # Verify Python and required packages
    if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
        error_log "Python 3.8+ required"
        exit 1
    fi
}

# Function to set environment variables
set_environment() {
    log "Setting up environment..."
    
    # Set environment for production
    export ENVIRONMENT="production"
    export REDIS_ENABLED="false"  # Disable Redis for cron jobs
    
    # Source .env file if it exists
    if [ -f "$BACKEND_DIR/.env" ]; then
        log "Loading environment variables from .env file"
        # shellcheck source=/dev/null
        source "$BACKEND_DIR/.env"
    else
        log "Warning: .env file not found. Make sure environment variables are set."
    fi
    
    # Verify required environment variables
    required_vars=("SUPABASE_URL" "SUPABASE_KEY")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            error_log "Required environment variable $var is not set"
            exit 1
        fi
    done
}

# Function to run automation task
run_automation() {
    local task="$1"
    log "Starting automation task: $task"
    
    cd "$BACKEND_DIR"
    
    case "$task" in
        "full")
            log "Running full automation pipeline..."
            if python scripts/daily_automation.py --full --save-results; then
                log "✅ Full automation completed successfully"
                return 0
            else
                error_log "❌ Full automation failed"
                return 1
            fi
            ;;
        "data")
            log "Running data ingestion..."
            if python scripts/daily_automation.py --data-ingestion --save-results; then
                log "✅ Data ingestion completed successfully"
                return 0
            else
                error_log "❌ Data ingestion failed"
                return 1
            fi
            ;;
        "predictions")
            log "Running predictions generation..."
            if python scripts/daily_automation.py --predictions --save-results; then
                log "✅ Predictions generation completed successfully"
                return 0
            else
                error_log "❌ Predictions generation failed"
                return 1
            fi
            ;;
        "health")
            log "Running health check..."
            if python scripts/daily_automation.py --health-check; then
                log "✅ Health check completed successfully"
                return 0
            else
                error_log "❌ Health check failed"
                return 1
            fi
            ;;
        *)
            error_log "Invalid task: $task. Valid options: full, data, predictions, health"
            return 1
            ;;
    esac
}

# Function to cleanup old logs (keep last 30 days)
cleanup_logs() {
    log "Cleaning up old log files..."
    find "$LOG_DIR" -name "automation_*.log" -type f -mtime +30 -delete 2>/dev/null || true
    find "$LOG_DIR" -name "automation_errors_*.log" -type f -mtime +30 -delete 2>/dev/null || true
}

# Function to send notification (if configured)
send_notification() {
    local status="$1"
    local message="$2"
    
    # Example: Send email notification (uncomment and configure as needed)
    # if command -v mail >/dev/null 2>&1; then
    #     echo "$message" | mail -s "Crypto Automation: $status" admin@example.com
    # fi
    
    # Example: Send to webhook (uncomment and configure as needed)
    # if [ -n "${WEBHOOK_URL:-}" ]; then
    #     curl -X POST "$WEBHOOK_URL" \
    #          -H "Content-Type: application/json" \
    #          -d "{\"status\":\"$status\",\"message\":\"$message\",\"timestamp\":\"$(date -Iseconds)\"}" \
    #          >/dev/null 2>&1 || true
    # fi
    
    log "Notification sent: $status - $message"
}

# Main execution
main() {
    local task="${1:-full}"
    local start_time
    local end_time
    local duration
    local exit_code=0
    
    start_time=$(date +%s)
    
    log "🚀 Starting crypto automation (task: $task)"
    log "Script: $0"
    log "Working directory: $(pwd)"
    log "Log file: $LOG_FILE"
    
    # Setup
    check_venv
    activate_venv
    set_environment
    
    # Run automation task
    if run_automation "$task"; then
        log "✅ Automation completed successfully"
        send_notification "SUCCESS" "Automation task '$task' completed successfully"
    else
        error_log "❌ Automation failed"
        send_notification "FAILURE" "Automation task '$task' failed. Check logs: $LOG_FILE"
        exit_code=1
    fi
    
    # Cleanup
    cleanup_logs
    
    # Calculate duration
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    log "📊 Automation summary:"
    log "  Task: $task"
    log "  Duration: ${duration}s"
    log "  Exit code: $exit_code"
    log "  Log file: $LOG_FILE"
    
    if [ $exit_code -ne 0 ] && [ -s "$ERROR_LOG" ]; then
        log "  Error log: $ERROR_LOG"
    fi
    
    exit $exit_code
}

# Show usage if no arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <task>"
    echo ""
    echo "Tasks:"
    echo "  full         - Run complete automation pipeline"
    echo "  data         - Run data ingestion only"
    echo "  predictions  - Run predictions generation only"
    echo "  health       - Run health check only"
    echo ""
    echo "Example cron entries:"
    echo "  # Daily automation at 6:00 AM"
    echo "  0 6 * * * $0 full"
    echo ""
    echo "  # Data ingestion every 6 hours"
    echo "  0 */6 * * * $0 data"
    echo ""
    echo "  # Predictions every 12 hours"
    echo "  0 */12 * * * $0 predictions"
    exit 1
fi

# Run main function
main "$@" 