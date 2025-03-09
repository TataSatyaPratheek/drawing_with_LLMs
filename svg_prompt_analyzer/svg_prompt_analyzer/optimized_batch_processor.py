"""
Optimized Batch Processor
=====================
This module provides high-performance batch processing for SVG generation
with dynamic batching, prioritization, and efficient resource management.
"""

import os
import time
import logging
import heapq
import queue
import threading
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Tuple, Optional, Callable, Union, Set, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type


class BatchItem(Generic[T, R]):
    """A single item in a processing batch with priority handling."""
    
    def __init__(self, 
                 item_id: str, 
                 input_data: T, 
                 priority: int = 0,
                 max_retries: int = 3):
        """
        Initialize a batch item.
        
        Args:
            item_id: Unique identifier for the item
            input_data: Input data to process
            priority: Processing priority (higher = higher priority)
            max_retries: Maximum number of retry attempts
        """
        self.item_id = item_id
        self.input_data = input_data
        self.priority = priority
        self.max_retries = max_retries
        self.result: Optional[R] = None
        self.error: Optional[Exception] = None
        self.attempts = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.status = "pending"  # pending, processing, completed, failed
        self.processing_time: Optional[float] = None
        
    def __lt__(self, other: 'BatchItem') -> bool:
        """Compare items based on priority for queue ordering."""
        if not isinstance(other, BatchItem):
            return NotImplemented
        return self.priority > other.priority  # Higher priority items come first
        
    def mark_start(self) -> None:
        """Mark the start of processing this item."""
        self.start_time = time.time()
        self.status = "processing"
        self.attempts += 1
        
    def mark_complete(self, result: R) -> None:
        """Mark the completion of processing this item."""
        self.end_time = time.time()
        self.result = result
        self.status = "completed"
        self.processing_time = self.end_time - (self.start_time or self.end_time)
        
    def mark_failed(self, error: Exception) -> None:
        """Mark the failure of processing this item."""
        self.end_time = time.time()
        self.error = error
        self.status = "failed"
        self.processing_time = self.end_time - (self.start_time or self.end_time)
        
    def can_retry(self) -> bool:
        """Check if the item can be retried."""
        return self.status == "failed" and self.attempts < self.max_retries
        
    def reset_for_retry(self) -> None:
        """Reset the item for retry."""
        self.start_time = None
        self.end_time = None
        self.status = "pending"
        self.processing_time = None


class BatchProcessor(Generic[T, R]):
    """
    High-performance batch processor with dynamic batching and resource management.
    """
    
    def __init__(self, 
                 process_func: Callable[[List[T]], List[R]],
                 optimal_batch_size: int = 8,
                 max_batch_size: int = 16,
                 min_batch_size: int = 1,
                 batch_timeout: float = 0.1,
                 max_workers: int = 4,
                 adaptive_batching: bool = True,
                 prefetch_next_batch: bool = True,
                 monitor_memory: bool = True,
                 memory_manager = None):
        """
        Initialize the batch processor.
        
        Args:
            process_func: Function to process a batch of items
            optimal_batch_size: Optimal batch size for efficiency
            max_batch_size: Maximum batch size
            min_batch_size: Minimum batch size
            batch_timeout: Maximum time to wait for a full batch (seconds)
            max_workers: Maximum number of worker threads
            adaptive_batching: Dynamically adjust batch size based on performance
            prefetch_next_batch: Prefetch the next batch while processing the current one
            monitor_memory: Monitor memory usage and adjust batch size accordingly
            memory_manager: Optional memory manager instance
        """
        self.process_func = process_func
        self.optimal_batch_size = optimal_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.batch_timeout = batch_timeout
        self.max_workers = max_workers
        self.adaptive_batching = adaptive_batching
        self.prefetch_next_batch = prefetch_next_batch
        self.monitor_memory = monitor_memory
        self.memory_manager = memory_manager
        
        # Queues for items
        self.input_queue: queue.PriorityQueue[BatchItem] = queue.PriorityQueue()
        self.output_queue: queue.Queue[BatchItem] = queue.Queue()
        self.retry_queue: queue.PriorityQueue[BatchItem] = queue.PriorityQueue()
        
        # Processing state
        self.processing = False
        self.processed_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.retry_count = 0
        
        # Performance tracking
        self.batch_times: List[float] = []
        self.item_times: List[float] = []
        self.current_batch_size = optimal_batch_size
        
        # Thread pool
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Processing thread
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # Prefetch thread and queue
        self.prefetch_thread = None
        self.prefetch_queue: queue.Queue[List[BatchItem]] = queue.Queue(maxsize=1)
        
        # Item tracking
        self.items_by_id: Dict[str, BatchItem] = {}
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
    def start(self) -> None:
        """Start the batch processor."""
        if self.processing:
            return
            
        self.processing = True
        self.stop_event.clear()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="BatchProcessingThread"
        )
        self.processing_thread.start()
        
        # Start prefetch thread if enabled
        if self.prefetch_next_batch:
            self.prefetch_thread = threading.Thread(
                target=self._prefetch_loop,
                daemon=True,
                name="BatchPrefetchThread"
            )
            self.prefetch_thread.start()
            
        logger.info(f"Batch processor started with batch size {self.current_batch_size}")
        
    def stop(self, wait_complete: bool = True) -> None:
        """
        Stop the batch