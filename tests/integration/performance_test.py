import time
import pytest
import os
import psutil
from src.presentation.dependencies.providers import AppContainer
from src.application.commands.image_uploaded_command import ImageUploadedCommand

def test_pipeline_execution_speed_benchmark(synthetic_image_bytes):
    """
    Benchmarks the execution speed of the Stone Color Detection pipeline.
    Verifies that the entire event-driven processing runs within the high-throughput
    threshold (<100ms per image) thanks to early resizing, vectorized skin masking,
    and random pixel sampling.
    """
    container = AppContainer()
    command_handler = container.command_handler

    # Warm-up run to exclude cold-start caching/import latencies
    command_warmup = ImageUploadedCommand(
        analysis_id="warmup",
        image_bytes=synthetic_image_bytes,
        filename="warmup.png"
    )
    command_handler.handle(command_warmup)

    # 1. Benchmark Execution Speed
    num_runs = 5
    latencies = []
    
    for i in range(num_runs):
        cmd = ImageUploadedCommand(
            analysis_id=f"bench-{i}",
            image_bytes=synthetic_image_bytes,
            filename=f"bench-{i}.png"
        )
        
        start_time = time.perf_counter()
        command_handler.handle(cmd)
        elapsed = time.perf_counter() - start_time
        latencies.append(elapsed)

    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000.0
    print(f"\nAverage Stone Color Analysis Pipeline Latency: {avg_latency_ms:.2f} ms")

    # Assert that average latency is well below 100 milliseconds
    assert avg_latency_ms < 100.0, f"Pipeline latency is too slow: {avg_latency_ms:.2f} ms"


def test_pipeline_memory_consumption(synthetic_image_bytes):
    """
    Validates that the pipeline does not cause memory leaks or excessive consumption
    by checking process RSS memory before and after runs.
    """
    process = psutil.Process(os.getpid())
    mem_before_kb = process.memory_info().rss / 1024.0

    container = AppContainer()
    command_handler = container.command_handler

    # Process multiple images in a loop
    for i in range(10):
        cmd = ImageUploadedCommand(
            analysis_id=f"memcheck-{i}",
            image_bytes=synthetic_image_bytes,
            filename=f"memcheck-{i}.png"
        )
        command_handler.handle(cmd)

    mem_after_kb = process.memory_info().rss / 1024.0
    delta_mem_kb = mem_after_kb - mem_before_kb
    print(f"\nMemory utilization delta after 10 runs: {delta_mem_kb:.2f} KB")
    
    # Assert that memory growth is minimal/controlled
    # A threshold of 5MB is extremely safe and ensures no leaks
    assert delta_mem_kb < 5120.0, f"Excessive memory consumption detected: {delta_mem_kb:.2f} KB"
