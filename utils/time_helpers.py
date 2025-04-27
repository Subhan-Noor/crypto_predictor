"""
Utility functions for date and time operations.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Union
import pandas as pd
import pytz


def convert_timestamp(timestamp: Union[str, int, float, datetime], 
                     to_timezone: Optional[str] = None) -> datetime:
    """
    Convert various timestamp formats to datetime.
    
    Args:
        timestamp: Timestamp in various formats
        to_timezone: Target timezone (e.g., 'UTC', 'America/New_York')
        
    Returns:
        Datetime object
    """
    # Convert string to datetime
    if isinstance(timestamp, str):
        try:
            dt = pd.to_datetime(timestamp)
        except:
            raise ValueError(f"Unable to parse timestamp: {timestamp}")
    # Convert Unix timestamp to datetime
    elif isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp)
    # Already datetime
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        raise TypeError(f"Unsupported timestamp type: {type(timestamp)}")
    
    # Convert timezone if specified
    if to_timezone:
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        dt = dt.astimezone(pytz.timezone(to_timezone))
    
    return dt


def get_date_range(start_date: Union[str, datetime], 
                  end_date: Union[str, datetime, None] = None,
                  days: Optional[int] = None) -> List[datetime]:
    """
    Get a list of dates in a given range.
    
    Args:
        start_date: Start date
        end_date: End date (optional)
        days: Number of days (if end_date not specified)
        
    Returns:
        List of datetime objects
    """
    # Convert start_date to datetime if it's a string
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
        
    # Determine end_date
    if end_date is None and days is not None:
        end_date = start_date + timedelta(days=days)
    elif end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Generate date range
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)
        
    return date_list


def time_ago(days: int = 0, hours: int = 0, minutes: int = 0) -> datetime:
    """
    Get datetime from N days/hours/minutes ago.
    
    Args:
        days: Number of days ago
        hours: Number of hours ago
        minutes: Number of minutes ago
        
    Returns:
        Datetime object for the specified time ago
    """
    now = datetime.now()
    delta = timedelta(days=days, hours=hours, minutes=minutes)
    return now - delta


def humanize_time_delta(timestamp: Union[str, datetime]) -> str:
    """
    Convert timestamp to human-readable relative time (e.g., "2 hours ago").
    
    Args:
        timestamp: Timestamp to convert
        
    Returns:
        Human-readable string
    """
    if isinstance(timestamp, str):
        dt = pd.to_datetime(timestamp)
    else:
        dt = timestamp
        
    now = datetime.now()
    
    if dt.tzinfo is not None:
        now = now.replace(tzinfo=dt.tzinfo)
        
    delta = now - dt
    
    # Convert to appropriate unit
    seconds = delta.total_seconds()
    
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        return f"{int(seconds // 60)} minutes ago"
    elif seconds < 86400:
        return f"{int(seconds // 3600)} hours ago"
    elif seconds < 604800:
        return f"{int(seconds // 86400)} days ago"
    elif seconds < 2592000:
        return f"{int(seconds // 604800)} weeks ago"
    elif seconds < 31536000:
        return f"{int(seconds // 2592000)} months ago"
    else:
        return f"{int(seconds // 31536000)} years ago" 