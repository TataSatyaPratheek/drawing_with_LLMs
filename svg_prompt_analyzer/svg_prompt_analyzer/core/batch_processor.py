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
                 memory_manager=None,
                 config: Optional[Dict[str, Any]] = None):
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
            config: Optional configuration dictionary to override defaults
        """
        # Apply configuration overrides if provided
        if config:
            batch_config = config.get("system", {}).get("batch_processing", {})
            optimal_batch_size = batch_config.get("optimal_batch_size", optimal_batch_size)
            max_batch_size = batch_config.get("max_batch_size", max_batch_size)
            min_batch_size = batch_config.get("min_batch_size", min_batch_size)
            batch_timeout = batch_config.get("batch_timeout", batch_timeout)
            max_workers = batch_config.get("max_workers", max_workers)
            adaptive_batching = batch_config.get("adaptive_batching", adaptive_batching)
            prefetch_next_batch = batch_config.get("prefetch_next_batch", prefetch_next_batch)
            monitor_memory = batch_config.get("monitor_memory", monitor_memory)
            
        self.process_func = process_func
        self.optimal_batch_size = optimal_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.batch_timeout = batch_timeout
        self.max_workers = max(1, min(max_workers, os.cpu_count() or 4))
        self.adaptive_batching = adaptive_batching
        self.prefetch_next_batch = prefetch_next_batch
        self.monitor_memory = monitor_memory
        self.memory_manager = memory_manager
        
        # Queues for items
        self.input_queue: 'queue.PriorityQueue[BatchItem]' = queue.PriorityQueue()
        self.output_queue: 'queue.Queue[BatchItem]' = queue.Queue()
        self.retry_queue: 'queue.PriorityQueue[BatchItem]' = queue.PriorityQueue()
        
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
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Processing thread
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # Prefetch thread and queue
        self.prefetch_thread = None
        self.prefetch_queue: 'queue.Queue[List[BatchItem]]' = queue.Queue(maxsize=1)
        
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
        Stop the batch processor.
        
        Args:
            wait_complete: Whether to wait for all items to finish processing
        """
        if not self.processing:
            return
            
        logger.info("Stopping batch processor")
        
        # Signal threads to stop
        self.stop_event.set()
        
        # Wait for processing to complete if requested
        if wait_complete:
            if self.processing_thread:
                self.processing_thread.join(timeout=10.0)
                
            if self.prefetch_thread:
                self.prefetch_thread.join(timeout=5.0)
        
        # Set state
        self.processing = False
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=wait_complete)
        
        logger.info("Batch processor stopped")
    
    def add_item(self, item_id: str, input_data: T, priority: int = 0) -> None:
        """
        Add an item to the processing queue.
        
        Args:
            item_id: Unique identifier for the item
            input_data: Input data to process
            priority: Processing priority (higher = higher priority)
        """
        item = BatchItem(item_id, input_data, priority)
        
        with self.lock:
            # Store item for tracking
            self.items_by_id[item_id] = item
            
            # Add to queue
            self.input_queue.put(item)
            
        logger.debug(f"Added item {item_id} to batch processing queue with priority {priority}")
    
    def add_items(self, items: List[Tuple[str, T, int]]) -> None:
        """
        Add multiple items to the processing queue.
        
        Args:
            items: List of (item_id, input_data, priority) tuples
        """
        with self.lock:
            for item_id, input_data, priority in items:
                item = BatchItem(item_id, input_data, priority)
                
                # Store item for tracking
                self.items_by_id[item_id] = item
                
                # Add to queue
                self.input_queue.put(item)
                
        logger.debug(f"Added {len(items)} items to batch processing queue")
    
    def get_result(self, item_id: str, timeout: Optional[float] = None) -> Optional[R]:
        """
        Get the result for a specific item.
        
        Args:
            item_id: Item identifier
            timeout: Maximum time to wait for result
            
        Returns:
            Result or None if not available
        """
        with self.lock:
            if item_id not in self.items_by_id:
                return None
                
            item = self.items_by_id[item_id]
            
        # If item is already completed, return result
        if item.status == "completed":
            return item.result
            
        # If item is failed, return None
        if item.status == "failed" and not item.can_retry():
            return None
            
        # Wait for completion
        start_time = time.time()
        while timeout is None or time.time() - start_time < timeout:
            with self.lock:
                item = self.items_by_id.get(item_id)
                if not item:
                    return None
                    
                if item.status == "completed":
                    return item.result
                    
                if item.status == "failed" and not item.can_retry():
                    return None
                    
            # Sleep briefly to avoid CPU spinning
            time.sleep(0.01)
            
        # Timeout reached
        return None
    
    def get_results(self, timeout: Optional[float] = None) -> Dict[str, R]:
        """
        Get all available results.
        
        Args:
            timeout: Maximum time to wait for all results
            
        Returns:
            Dictionary mapping item IDs to results
        """
        results = {}
        
        # Check if any items are still processing
        start_time = time.time()
        pending_items = True
        
        while pending_items and (timeout is None or time.time() - start_time < timeout):
            pending_items = False
            
            with self.lock:
                for item_id, item in self.items_by_id.items():
                    if item.status == "completed":
                        results[item_id] = item.result
                    elif item.status in ["pending", "processing"] or (item.status == "failed" and item.can_retry()):
                        pending_items = True
                        
            if pending_items:
                # Sleep briefly to avoid CPU spinning
                time.sleep(0.01)
                
        # Add any additional results that completed during the last check
        with self.lock:
            for item_id, item in self.items_by_id.items():
                if item.status == "completed":
                    results[item_id] = item.result
                    
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self.lock:
            stats = {
                "processed_count": self.processed_count,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "retry_count": self.retry_count,
                "pending_count": self.input_queue.qsize(),
                "current_batch_size": self.current_batch_size
            }
            
            # Calculate average processing times if available
            if self.batch_times:
                stats["avg_batch_time"] = sum(self.batch_times) / len(self.batch_times)
                
            if self.item_times:
                stats["avg_item_time"] = sum(self.item_times) / len(self.item_times)
                
            return stats
    
    def _processing_loop(self) -> None:
        """Main processing loop for batch processing."""
        logger.info("Starting batch processing loop")
        
        while not self.stop_event.is_set():
            try:
                # Prepare batch
                batch = self._prepare_batch()
                
                if not batch:
                    # No items to process, sleep briefly
                    time.sleep(0.01)
                    continue
                    
                # Process batch
                batch_size = len(batch)
                logger.debug(f"Processing batch of {batch_size} items")
                
                # Track batch processing time
                batch_start_time = time.time()
                
                # Extract input data
                input_data = [item.input_data for item in batch]
                
                # Mark items as processing
                for item in batch:
                    item.mark_start()
                
                # Process batch
                try:
                    results = self.process_func(input_data)
                    
                    # Check results length
                    if len(results) != len(batch):
                        logger.error(f"Result length mismatch: got {len(results)}, expected {len(batch)}")
                        # Mark all items as failed
                        for item in batch:
                            item.mark_failed(ValueError("Result length mismatch"))
                            self.retry_queue.put(item)
                            
                        self.failure_count += len(batch)
                        continue
                        
                    # Handle results
                    for i, result in enumerate(results):
                        item = batch[i]
                        
                        # Mark as complete
                        item.mark_complete(result)
                        
                        # Add to output queue
                        self.output_queue.put(item)
                        
                    # Update statistics
                    self.success_count += len(batch)
                    
                except Exception as e:
                    logger.error(f"Batch processing error: {str(e)}")
                    
                    # Mark all items as failed
                    for item in batch:
                        item.mark_failed(e)
                        
                        # Add to retry queue if retries available
                        if item.can_retry():
                            item.reset_for_retry()
                            self.retry_queue.put(item)
                            self.retry_count += 1
                        else:
                            # Add to output queue as failed
                            self.output_queue.put(item)
                            self.failure_count += 1
                
                # Update statistics
                self.processed_count += len(batch)
                batch_time = time.time() - batch_start_time
                self.batch_times.append(batch_time)
                
                # Keep only recent times for adaptive sizing
                if len(self.batch_times) > 10:
                    self.batch_times = self.batch_times[-10:]
                    
                # Calculate per-item time
                item_time = batch_time / len(batch)
                self.item_times.extend([item_time] * len(batch))
                
                # Keep only recent times for adaptive sizing
                if len(self.item_times) > 100:
                    self.item_times = self.item_times[-100:]
                
                # Update batch size if adaptive batching is enabled
                if self.adaptive_batching and len(self.batch_times) >= 3:
                    self._adapt_batch_size()
                    
                logger.debug(f"Batch processed in {batch_time:.3f}s ({item_time:.3f}s per item)")
                
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                time.sleep(0.1)  # Sleep before retrying
                
        logger.info("Batch processing loop stopped")
    
    def _prepare_batch(self) -> List[BatchItem]:
        """
        Prepare a batch of items for processing.
        
        Returns:
            List of batch items to process
        """
        # Use prefetched batch if available
        if self.prefetch_next_batch and not self.prefetch_queue.empty():
            try:
                return self.prefetch_queue.get_nowait()
            except queue.Empty:
                pass
                
        # Check memory if monitoring is enabled
        current_batch_size = self.current_batch_size
        if self.monitor_memory and self.memory_manager:
            # Adjust batch size based on memory availability
            item_size = 10 * 1024 * 1024  # Default 10MB per item
            if hasattr(self.memory_manager, 'calculate_optimal_batch_size'):
                optimal_size = self.memory_manager.calculate_optimal_batch_size(
                    item_size_estimate=item_size,
                    model_size_estimate=0,
                    target_device="auto"
                )
                current_batch_size = min(current_batch_size, optimal_size)
        
        # Get items from retry queue first
        retry_items = []
        while len(retry_items) < current_batch_size and not self.retry_queue.empty():
            try:
                retry_items.append(self.retry_queue.get_nowait())
            except queue.Empty:
                break
                
        # Get remaining items from input queue
        remaining_slots = current_batch_size - len(retry_items)
        regular_items = []
        
        if remaining_slots > 0:
            # Use batch timeout only if we have some items but not a full batch
            timeout = None
            
            try:
                # Get first item with timeout
                item = self.input_queue.get(timeout=self.batch_timeout)
                regular_items.append(item)
                remaining_slots -= 1
                
                # Get remaining items with shorter timeout
                while remaining_slots > 0 and not self.input_queue.empty():
                    try:
                        item = self.input_queue.get_nowait()
                        regular_items.append(item)
                        remaining_slots -= 1
                    except queue.Empty:
                        break
                        
            except queue.Empty:
                # No items available
                pass
                
        # Combine retry and regular items
        batch = retry_items + regular_items
        
        return batch
    
    def _prefetch_loop(self) -> None:
        """Prefetch loop to prepare next batch while current batch is processing."""
        logger.info("Starting batch prefetch loop")
        
        while not self.stop_event.is_set():
            try:
                # Skip if prefetch queue is full
                if self.prefetch_queue.full():
                    time.sleep(0.01)
                    continue
                    
                # Prepare next batch
                batch = self._prepare_batch()
                
                if batch:
                    # Add to prefetch queue
                    self.prefetch_queue.put(batch)
                else:
                    # No items to prefetch, sleep briefly
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in prefetch loop: {str(e)}")
                time.sleep(0.1)  # Sleep before retrying
                
        logger.info("Batch prefetch loop stopped")
    
    def _adapt_batch_size(self) -> None:
        """Adapt batch size based on processing performance."""
        # Calculate average batch processing time
        avg_time = sum(self.batch_times) / len(self.batch_times)
        
        # Current items per second
        current_ips = self.current_batch_size / avg_time
        
        # Calculate potential new batch sizes
        increase_size = min(self.current_batch_size + 2, self.max_batch_size)
        decrease_size = max(self.current_batch_size - 1, self.min_batch_size)
        
        # Estimate performance with new sizes
        # This is a simple linear estimation and could be improved with more data
        estimated_increase_time = avg_time * (increase_size / self.current_batch_size)
        estimated_increase_ips = increase_size / estimated_increase_time
        
        estimated_decrease_time = avg_time * (decrease_size / self.current_batch_size)
        estimated_decrease_ips = decrease_size / estimated_decrease_time
        
        # Choose new batch size
        if estimated_increase_ips > current_ips and estimated_increase_ips > estimated_decrease_ips:
            # Increasing batch size improves throughput
            self.current_batch_size = increase_size
            logger.info(f"Increased batch size to {self.current_batch_size} for better throughput")
        elif estimated_decrease_ips > current_ips:
            # Decreasing batch size improves throughput
            self.current_batch_size = decrease_size
            logger.info(f"Decreased batch size to {self.current_batch_size} for better throughput")