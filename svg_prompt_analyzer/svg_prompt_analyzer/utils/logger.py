"""
Production-grade logging system with performance optimizations.
Supports structured logging, rotation, and multiple output formats.
"""

import os
import sys
import time
import json
import atexit
import logging
import traceback
from typing import Dict, Any, List, Optional, Union, Callable
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
import queue
from datetime import datetime
from functools import wraps
import threading
import socket

# Import core optimizations - FIX: Use package-relative import
from svg_prompt_analyzer.core import CONFIG, get_thread_pool

# Constants
DEFAULT_LOG_LEVEL = logging.INFO
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
JSON_LOG_FORMAT = True  # Use JSON for structured logging in production

# Global variables
_log_queue = None
_queue_listener = None
_thread_local = threading.local()
_log_metrics = {
    'total_logs': 0,
    'error_logs': 0,
    'warn_logs': 0,
    'start_time': time.time()
}
_lock = threading.RLock()


class ContextualFilter(logging.Filter):
    """Add contextual information to log records."""
    
    def __init__(self):
        super().__init__()
        self.hostname = socket.gethostname()
        
    def filter(self, record):
        # Add host information
        record.hostname = self.hostname
        
        # Add thread context if available
        if hasattr(_thread_local, 'context'):
            for key, value in _thread_local.context.items():
                setattr(record, key, value)
        
        # Add trace ID if available
        if not hasattr(record, 'trace_id') and hasattr(_thread_local, 'trace_id'):
            record.trace_id = _thread_local.trace_id
            
        # Count logs by type
        with _lock:
            _log_metrics['total_logs'] += 1
            if record.levelno >= logging.ERROR:
                _log_metrics['error_logs'] += 1
            elif record.levelno >= logging.WARNING:
                _log_metrics['warn_logs'] += 1
                
        return True


class JsonFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'line': record.lineno,
            'process': record.process,
            'thread': record.thread,
            'hostname': getattr(record, 'hostname', 'unknown')
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
            
        # Add custom attributes
        for key, value in record.__dict__.items():
            if key not in ('args', 'asctime', 'created', 'exc_info', 'exc_text', 
                          'filename', 'funcName', 'id', 'levelname', 'levelno',
                          'lineno', 'module', 'msecs', 'message', 'msg', 'name',
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'stack_info', 'thread', 'threadName', 'hostname'):
                log_data[key] = value
                
        return json.dumps(log_data)


def setup_logging(
    app_name: str = 'app',
    log_dir: Optional[str] = None,
    log_level: int = DEFAULT_LOG_LEVEL,
    use_json: Optional[bool] = None,
    log_to_console: bool = True,
    log_to_file: bool = True,
    async_logging: bool = True
) -> logging.Logger:
    """
    Configure production-grade logging with performance optimizations.
    
    Args:
        app_name: Application name for logger
        log_dir: Directory to store log files
        log_level: Log level
        use_json: Use JSON format for logs
        log_to_console: Also log to console
        log_to_file: Log to file
        async_logging: Use asynchronous logging for better performance
        
    Returns:
        Configured logger
    """
    global _log_queue, _queue_listener
    
    # Default log directory
    if log_dir is None:
        log_dir = os.environ.get('LOG_DIR', os.path.join(os.getcwd(), 'logs'))
        
    # Create log directory if it doesn't exist
    if log_to_file and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    # Use JSON in production unless explicitly disabled
    if use_json is None:
        use_json = JSON_LOG_FORMAT
        
    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(log_level)
    logger.handlers = []  # Remove existing handlers
    
    # Create contextual filter
    context_filter = ContextualFilter()
    logger.addFilter(context_filter)
    
    # Create handlers
    handlers = []
    
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        handlers.append(console_handler)
        
    if log_to_file:
        log_file = os.path.join(log_dir, f"{app_name}.log")
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=MAX_LOG_SIZE, 
            backupCount=BACKUP_COUNT
        )
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
        
    # Create and set formatters
    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(LOG_FORMAT)
        
    # Apply formatters to handlers
    for handler in handlers:
        handler.setFormatter(formatter)
        
    # Set up asynchronous logging if enabled
    if async_logging:
        _log_queue = queue.Queue()
        queue_handler = QueueHandler(_log_queue)
        logger.addHandler(queue_handler)
        
        # Start queue listener
        _queue_listener = QueueListener(_log_queue, *handlers, respect_handler_level=True)
        _queue_listener.start()
        
        # Register cleanup
        atexit.register(lambda: _queue_listener.stop() if _queue_listener else None)
    else:
        # Add handlers directly
        for handler in handlers:
            logger.addHandler(handler)
    
    return logger


def setup_logger(
    level: str = None,
    log_file: Optional[str] = None,
    console: bool = True,
    format_str: Optional[str] = None
) -> None:
    """
    Configure the root logger for the application.
    
    Args:
        level: Logging level
        log_file: Optional file to log to
        console: Whether to log to console
        format_str: Optional custom format string
    """
    # Get log level
    level_value = getattr(logging, level.upper(), logging.INFO) if level else logging.INFO
    
    # Get format string
    format_str = format_str or LOG_FORMAT
    formatter = logging.Formatter(format_str)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level_value)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level_value)
        root_logger.addHandler(console_handler)
    
    # Add file handler if provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level_value)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def with_context(**context):
    """
    Decorator to add context to logs within a function.
    
    Args:
        **context: Context key-value pairs
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            old_context = getattr(_thread_local, 'context', {}).copy()
            if not hasattr(_thread_local, 'context'):
                _thread_local.context = {}
                
            # Update with new context
            _thread_local.context.update(context)
            
            try:
                return func(*args, **kwargs)
            finally:
                # Restore old context
                _thread_local.context = old_context
                
        return wrapper
    return decorator


def set_trace_id(trace_id: str = None) -> str:
    """
    Set a trace ID for the current thread.
    
    Args:
        trace_id: Trace ID to set or None to generate a new one
        
    Returns:
        The trace ID
    """
    if trace_id is None:
        import uuid
        trace_id = str(uuid.uuid4())
        
    _thread_local.trace_id = trace_id
    return trace_id


def get_trace_id() -> Optional[str]:
    """Get the current trace ID."""
    return getattr(_thread_local, 'trace_id', None)


def clear_trace_id() -> None:
    """Clear the current trace ID."""
    if hasattr(_thread_local, 'trace_id'):
        delattr(_thread_local, 'trace_id')


class LogCapture:
    """Context manager to capture logs for testing or analysis."""
    
    def __init__(self, logger_name: str = None, level: int = logging.DEBUG):
        self.logger_name = logger_name
        self.level = level
        self.handler = None
        self.logs = []
        
    def __enter__(self):
        # Create handler that captures logs
        class CaptureHandler(logging.Handler):
            def __init__(self, logs):
                super().__init__()
                self.logs = logs
                
            def emit(self, record):
                self.logs.append(self.format(record))
                
        # Set up handler
        self.handler = CaptureHandler(self.logs)
        self.handler.setLevel(self.level)
        self.handler.setFormatter(logging.Formatter(LOG_FORMAT))
        
        # Add handler to logger
        logger = logging.getLogger(self.logger_name)
        logger.addHandler(self.handler)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove handler
        if self.handler:
            logger = logging.getLogger(self.logger_name)
            logger.removeHandler(self.handler)
            self.handler = None


def log_function_call(logger=None, level=logging.DEBUG):
    """
    Decorator to log function calls with performance metrics.
    
    Args:
        logger: Logger to use or None to use function's module logger
        level: Log level
    """
    def decorator(func):
        # Get logger
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)
            
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            # Generate call ID
            call_id = f"{func_name}_{id(args)}"
            
            # Log call
            arg_str = ', '.join([
                str(a) if len(str(a)) < 100 else f"{str(a)[:97]}..." 
                for a in args
            ])
            kwargs_str = ', '.join([
                f"{k}={v if len(str(v)) < 100 else f'{str(v)[:97]}...'}" 
                for k, v in kwargs.items()
            ])
            logger.log(
                level, 
                f"CALL {call_id}: {func_name}({arg_str}{', ' if arg_str and kwargs_str else ''}{kwargs_str})"
            )
            
            # Measure execution time
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.log(
                    level, 
                    f"RETURN {call_id}: {func_name} completed in {elapsed:.6f}s"
                )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.exception(
                    f"ERROR {call_id}: {func_name} failed after {elapsed:.6f}s with {type(e).__name__}: {str(e)}"
                )
                raise
                
        return wrapper
    return decorator


def get_log_metrics() -> Dict[str, Any]:
    """Get metrics about logging activity."""
    with _lock:
        metrics = _log_metrics.copy()
        metrics['uptime'] = time.time() - metrics['start_time']
        metrics['logs_per_second'] = metrics['total_logs'] / metrics['uptime'] if metrics['uptime'] > 0 else 0
        return metrics


def log_exception(
    logger: logging.Logger,
    exc: Exception,
    level: int = logging.ERROR,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an exception with context.
    
    Args:
        logger: Logger to use
        exc: Exception to log
        level: Log level
        context: Additional context to log
    """
    # Build message
    message = f"Exception: {type(exc).__name__}: {str(exc)}"
    
    # Add context if provided
    if context:
        context_str = ', '.join(f"{k}={v}" for k, v in context.items())
        message += f" [Context: {context_str}]"
        
    # Log with traceback
    logger.log(level, message, exc_info=exc)


def rotate_logs(log_dir: Optional[str] = None) -> None:
    """
    Force log rotation, useful for maintenance jobs.
    
    Args:
        log_dir: Directory containing logs
    """
    # Default to current directory
    if log_dir is None:
        log_dir = os.environ.get('LOG_DIR', os.path.join(os.getcwd(), 'logs'))
        
    # Find all log files
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith('.log'):
                file_path = os.path.join(root, file)
                
                try:
                    # Open each log handler to trigger rotation if needed
                    for handler in logging.root.handlers:
                        if isinstance(handler, RotatingFileHandler):
                            if handler.baseFilename == file_path:
                                # Force rotation by setting file size to max
                                handler.doRollover()
                except Exception as e:
                    sys.stderr.write(f"Error rotating log {file_path}: {e}\n")


def setup_remote_logging(
    host: str,
    port: int,
    app_name: str,
    secure: bool = True
) -> logging.Logger:
    """
    Set up logging to a remote logging service (e.g., Logstash, Fluentd).
    
    Args:
        host: Remote host
        port: Remote port
        app_name: Application name
        secure: Use TLS
        
    Returns:
        Configured logger
    """
    try:
        # Try to import remote logging handlers
        from logging.handlers import SocketHandler, SysLogHandler
        
        # Create logger
        logger = logging.getLogger(app_name)
        logger.setLevel(logging.INFO)
        
        try:
            # Try socket handler first
            handler = SocketHandler(host, port)
            
            # Add TLS if requested
            if secure:
                import ssl
                ssl_context = ssl.create_default_context()
                handler.sock = ssl_context.wrap_socket(
                    handler.sock,
                    server_hostname=host
                )
        except:
            # Fall back to syslog
            handler = SysLogHandler(address=(host, port))
            
        # Configure handler
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        
        return logger
    except ImportError:
        sys.stderr.write("Remote logging requires socketserver and ssl modules\n")
        # Fall back to standard logging
        return setup_logging(app_name=app_name)