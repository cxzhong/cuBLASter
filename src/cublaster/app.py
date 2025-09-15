#!/usr/bin/env python3
"""
cuBLASter Application

CUDA-accelerated lattice reduction application based on BLASter.
This application provides similar functionality to BLASter but uses GPU acceleration.
"""

import sys
import argparse
import time
import math
import numpy as np
from typing import Optional, List, Tuple

try:
    import cublaster
    from cublaster import CudaLatticeReducer, get_memory_info
except ImportError as e:
    print(f"Error: Cannot import cuBLASter: {e}")
    print("Please build the package first with: pip install -e .")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="cuBLASter: CUDA-accelerated lattice reduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i lattice.txt -pq          # Reduce lattice with profile and quiet output
  %(prog)s -pvi lattice.txt            # Verbose reduction with profile
  %(prog)s -pq -d4                     # Deep-LLL with depth 4
  %(prog)s -pq -b60 -t1 -P2           # Progressive BKZ-60
  
Input formats:
  - Read from stdin if -i not specified
  - Lattice format: space-separated matrix rows
        """
    )
    
    # Input/Output options
    parser.add_argument('-i', '--input', metavar='FILE',
                       help='Input lattice file (default: stdin)')
    parser.add_argument('-o', '--output', metavar='FILE',
                       help='Output lattice file (default: stdout)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Quiet output (minimal logging)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('-p', '--profile', action='store_true',
                       help='Show profile of reduced lattice')
    
    # Algorithm options
    parser.add_argument('-d', '--depth', type=int, default=1,
                       help='Deep-LLL depth (default: 1)')
    parser.add_argument('-b', '--blocksize', type=int,
                       help='BKZ block size')
    parser.add_argument('-t', '--tours', type=int, default=1,
                       help='Number of BKZ tours (default: 1)')
    parser.add_argument('-P', '--progressive', type=int,
                       help='Progressive BKZ increment')
    
    # CUDA options
    parser.add_argument('--device', type=int, default=0,
                       help='CUDA device ID (default: 0)')
    parser.add_argument('--memory-info', action='store_true',
                       help='Show GPU memory information')
    
    return parser.parse_args()


def read_lattice(file_path: Optional[str] = None) -> np.ndarray:
    """Read lattice from file or stdin"""
    if file_path:
        with open(file_path, 'r') as f:
            content = f.read()
    else:
        content = sys.stdin.read()
    
    lines = content.strip().split('\n')
    matrix = []
    
    for line in lines:
        if line.strip():
            row = [int(x) for x in line.split()]
            matrix.append(row)
    
    return np.array(matrix, dtype=np.int64)


def write_lattice(lattice: np.ndarray, file_path: Optional[str] = None):
    """Write lattice to file or stdout"""
    output = []
    for row in lattice:
        output.append(' '.join(map(str, row)))
    
    result = '\n'.join(output) + '\n'
    
    if file_path:
        with open(file_path, 'w') as f:
            f.write(result)
    else:
        sys.stdout.write(result)


def compute_profile(lattice: np.ndarray) -> List[float]:
    """Compute the profile (log of norms) of lattice vectors"""
    profile = []
    for i in range(lattice.shape[0]):
        norm = np.linalg.norm(lattice[i].astype(float))
        if norm > 0:
            profile.append(math.log(norm, 10))
        else:
            profile.append(0.0)
    return profile


def compute_root_hermite_factor(lattice: np.ndarray) -> Tuple[float, float]:
    """Compute root Hermite factor and first vector norm"""
    first_norm = np.linalg.norm(lattice[0].astype(float))
    
    # Compute determinant (approximate for large matrices)
    det = abs(np.linalg.det(lattice.astype(float)))
    if det <= 0:
        return float('inf'), first_norm
    
    n = lattice.shape[0]
    root_det = det ** (1.0 / n)
    
    # Hermite factor
    hermite_factor = first_norm / root_det
    
    return hermite_factor, first_norm


def format_profile(profile: List[float], max_display: int = 10) -> str:
    """Format profile for display"""
    if len(profile) <= max_display:
        formatted = [f"{x:.2f}" for x in profile]
        return "[" + " ".join(formatted) + "]"
    else:
        start = [f"{x:.2f}" for x in profile[:max_display//2]]
        end = [f"{x:.2f}" for x in profile[-max_display//2:]]
        return "[" + " ".join(start) + " ... " + " ".join(end) + "]"


def main():
    """Main application entry point"""
    args = parse_arguments()
    
    # Show GPU memory info if requested
    if args.memory_info:
        try:
            mem_info = get_memory_info()
            print(f"GPU Memory: {mem_info['used_mb']}MB used / {mem_info['total_mb']}MB total "
                  f"({mem_info['free_mb']}MB free)")
        except Exception as e:
            print(f"Error getting GPU memory info: {e}")
            return 1
    
    try:
        # Read input lattice
        start_time = time.time()
        
        if not args.quiet:
            print("Reading lattice...")
        
        lattice = read_lattice(args.input)
        
        if not args.quiet:
            print(f"Lattice dimensions: {lattice.shape[0]} x {lattice.shape[1]}")
        
        # Initialize CUDA lattice reducer
        reducer = CudaLatticeReducer(device_id=args.device)
        
        if args.verbose:
            mem_info = reducer.get_memory_info()
            print(f"GPU Memory: {mem_info['used_mb']}MB used / {mem_info['total_mb']}MB total")
        
        # Perform reduction (placeholder - actual algorithms to be implemented)
        if not args.quiet:
            print("Performing lattice reduction on GPU...")
        
        reduction_start = time.time()
        reduced_lattice = reducer.reduce(lattice)  # This is currently a placeholder
        reduction_time = time.time() - reduction_start
        
        # Show profile if requested
        if args.profile:
            profile = compute_profile(reduced_lattice)
            hermite_factor, first_norm = compute_root_hermite_factor(reduced_lattice)
            
            print(f"Profile: {format_profile(profile)}")
            print(f"Root Hermite factor: {hermite_factor:.6f}, ∥b_1∥ = {first_norm:.3f}")
            
            if args.verbose:
                mem_info = reducer.get_memory_info()
                print(f"GPU Memory Used: {mem_info['used_mb']}MB")
        
        # Write output
        write_lattice(reduced_lattice, args.output)
        
        total_time = time.time() - start_time
        
        if args.verbose:
            print(f"Total time: {total_time:.3f}s")
            print(f"Reduction time: {reduction_time:.3f}s")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())