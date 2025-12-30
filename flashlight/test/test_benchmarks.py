"""Benchmark tests with plots."""
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ..Tensor import Tensor
from ..creators import randn
from ..neural import Linear, Tanh, NeuralNet
from ..Config import Config

class BenchmarkSuite:
    """Benchmark suite for Flashlight library."""
    
    def __init__(self, output_dir="benchmark_results"):
        """Initialize benchmark suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        Config.enable_backprop = True
    
    def benchmark_tensor_operations(self, sizes=[10, 50, 100, 500, 1000]):
        """Benchmark tensor operations at different sizes."""
        results = {
            'addition': [],
            'multiplication': [],
            'matrix_mult': [],
            'tanh': [],
            'sum': []
        }
        
        for size in sizes:
            print(f"Benchmarking size {size}...")
            
            # Addition
            a = randn(size, size)
            b = randn(size, size)
            start = time.time()
            _ = a + b
            results['addition'].append(time.time() - start)
            
            # Multiplication
            start = time.time()
            _ = a * b
            results['multiplication'].append(time.time() - start)
            
            # Matrix multiplication
            start = time.time()
            _ = a @ b
            results['matrix_mult'].append(time.time() - start)
            
            # Tanh
            start = time.time()
            _ = a.tanh()
            results['tanh'].append(time.time() - start)
            
            # Sum
            start = time.time()
            _ = a.sum()
            results['sum'].append(time.time() - start)
        
        self._plot_benchmark(results, sizes, "Tensor Operations", "tensor_operations.png")
        return results
    
    def benchmark_backward_pass(self, sizes=[10, 50, 100, 500, 1000]):
        """Benchmark backward pass at different sizes."""
        results = {
            'forward': [],
            'backward': []
        }
        
        for size in sizes:
            print(f"Benchmarking backward pass size {size}...")
            
            a = randn(size, size, requires_grad=True)
            b = randn(size, size, requires_grad=True)
            
            # Forward pass
            start = time.time()
            c = (a @ b).tanh().sum()
            forward_time = time.time() - start
            results['forward'].append(forward_time)
            
            # Backward pass
            start = time.time()
            c.backward()
            backward_time = time.time() - start
            results['backward'].append(backward_time)
        
        self._plot_benchmark(results, sizes, "Forward vs Backward Pass", "forward_backward.png")
        return results
    
    def benchmark_neural_network(self, hidden_sizes=[32, 64, 128, 256, 512], batch_size=32):
        """Benchmark neural network forward and backward passes."""
        results = {
            'forward': [],
            'backward': []
        }
        
        input_size = 784
        output_size = 10
        
        for hidden_size in hidden_sizes:
            print(f"Benchmarking network with hidden size {hidden_size}...")
            
            def calculate_output(x):
                return x
            
            def calculate_loss(output, target):
                diff = output - target
                return (diff * diff).mean()
            
            net = NeuralNet(calculate_output=calculate_output, calculate_loss=calculate_loss)
            net.add_layer(Linear(input_size, hidden_size))
            net.add_layer(Tanh())
            net.add_layer(Linear(hidden_size, hidden_size))
            net.add_layer(Tanh())
            net.add_layer(Linear(hidden_size, output_size))
            
            x = randn(batch_size, input_size)
            y = randn(batch_size, output_size)
            
            # Forward pass
            start = time.time()
            loss = net.forward(x, y)
            forward_time = time.time() - start
            results['forward'].append(forward_time)
            
            # Backward pass
            start = time.time()
            loss.backward()
            backward_time = time.time() - start
            results['backward'].append(backward_time)
        
        self._plot_benchmark(results, hidden_sizes, "Neural Network Performance", "neural_network.png")
        return results
    
    def benchmark_gradient_computation(self, sizes=[10, 50, 100, 500, 1000]):
        """Benchmark gradient computation complexity."""
        results = {
            'simple': [],
            'medium': [],
            'complex': []
        }
        
        for size in sizes:
            print(f"Benchmarking gradient computation size {size}...")
            
            # Simple: single operation
            a = randn(size, size, requires_grad=True)
            start = time.time()
            c = a.sum()
            c.backward()
            results['simple'].append(time.time() - start)
            
            # Medium: chain of operations
            a = randn(size, size, requires_grad=True)
            b = randn(size, size, requires_grad=True)
            start = time.time()
            c = (a @ b).tanh().sum()
            c.backward()
            results['medium'].append(time.time() - start)
            
            # Complex: many operations
            a = randn(size, size, requires_grad=True)
            b = randn(size, size, requires_grad=True)
            c = randn(size, size, requires_grad=True)
            start = time.time()
            d = ((a @ b).tanh() + c).exp().log().sum()
            d.backward()
            results['complex'].append(time.time() - start)
        
        self._plot_benchmark(results, sizes, "Gradient Computation Complexity", "gradient_complexity.png")
        return results
    
    def benchmark_memory_efficiency(self, num_iterations=[10, 50, 100, 500, 1000]):
        """Benchmark memory efficiency over many iterations."""
        import sys
        
        results = {
            'iterations': [],
            'memory_mb': []
        }
        
        for num_iter in num_iterations:
            print(f"Benchmarking memory efficiency with {num_iter} iterations...")
            
            # Clear any existing graphs
            Config.enable_backprop = True
            
            # Run many forward/backward passes
            a = randn(100, 100, requires_grad=True)
            b = randn(100, 100, requires_grad=True)
            
            start_memory = sys.getsizeof(a) + sys.getsizeof(b)
            
            for _ in range(num_iter):
                c = (a @ b).sum()
                c.backward()
                # Clear gradients for next iteration
                a.grad = None
                b.grad = None
            
            # Approximate memory usage
            memory_mb = (sys.getsizeof(a) + sys.getsizeof(b)) / (1024 * 1024)
            
            results['iterations'].append(num_iter)
            results['memory_mb'].append(memory_mb)
        
        self._plot_memory_benchmark(results, "Memory Efficiency", "memory_efficiency.png")
        return results
    
    def _plot_benchmark(self, results, x_values, title, filename):
        """Plot benchmark results."""
        plt.figure(figsize=(10, 6))
        
        for operation, times in results.items():
            plt.plot(x_values, times, marker='o', label=operation)
        
        plt.xlabel('Size / Hidden Size')
        plt.ylabel('Time (seconds)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {output_path}")
    
    def _plot_memory_benchmark(self, results, title, filename):
        """Plot memory benchmark results."""
        plt.figure(figsize=(10, 6))
        
        plt.plot(results['iterations'], results['memory_mb'], marker='o')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Memory Usage (MB)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {output_path}")
    
    def run_all_benchmarks(self):
        """Run all benchmarks."""
        print("=" * 50)
        print("Running Flashlight Benchmarks")
        print("=" * 50)
        
        results = {}
        
        print("\n1. Tensor Operations Benchmark")
        results['tensor_ops'] = self.benchmark_tensor_operations()
        
        print("\n2. Backward Pass Benchmark")
        results['backward'] = self.benchmark_backward_pass()
        
        print("\n3. Neural Network Benchmark")
        results['neural_net'] = self.benchmark_neural_network()
        
        print("\n4. Gradient Computation Benchmark")
        results['gradient'] = self.benchmark_gradient_computation()
        
        print("\n5. Memory Efficiency Benchmark")
        results['memory'] = self.benchmark_memory_efficiency()
        
        print("\n" + "=" * 50)
        print("All benchmarks completed!")
        print(f"Results saved to {self.output_dir}")
        print("=" * 50)
        
        return results

def run_benchmarks():
    """Convenience function to run all benchmarks."""
    suite = BenchmarkSuite()
    return suite.run_all_benchmarks()

if __name__ == "__main__":
    run_benchmarks()

