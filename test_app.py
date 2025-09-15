#!/usr/bin/env python3
"""
Test the cuBLASter application with the example lattice
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
import cublaster_core

def test_lattice_application():
    print("cuBLASter Application Test")
    print("=" * 50)
    
    # Initialize
    cublaster_core.init_cuda(0)
    mem_info = cublaster_core.get_memory_info()
    print(f"GPU Memory: {mem_info['used_mb']}MB used / {mem_info['total_mb']}MB total")
    
    # Read example lattice
    lattice_file = "examples/lattice_example.txt"
    if os.path.exists(lattice_file):
        with open(lattice_file, 'r') as f:
            lines = f.read().strip().split('\n')
        
        matrix = []
        for line in lines:
            if line.strip():
                row = [int(x) for x in line.split()]
                matrix.append(row)
        
        lattice = np.array(matrix, dtype=np.int64)
        print(f"Lattice dimensions: {lattice.shape[0]} x {lattice.shape[1]}")
        print("Input lattice:")
        print(lattice)
        
        # For now, just test that we can perform matrix operations on it
        # (Full lattice reduction algorithms would be implemented separately)
        
        # Test self-multiplication as a basic operation
        if lattice.shape[0] == lattice.shape[1]:  # Square matrix
            result = cublaster_core.matmul(lattice, lattice)
            print("\nSelf-multiplication result:")
            print(result)
            
        # Compute profile (log of norms)
        profile = []
        for i in range(lattice.shape[0]):
            norm = np.linalg.norm(lattice[i].astype(float))
            if norm > 0:
                profile.append(np.log10(norm))
            else:
                profile.append(0.0)
        
        print(f"\nProfile: {profile}")
        
        # Compute first vector norm
        first_norm = np.linalg.norm(lattice[0].astype(float))
        print(f"∥b_1∥ = {first_norm:.3f}")
        
        print("\n✓ Application test completed successfully")
        
    else:
        print(f"Example lattice file not found: {lattice_file}")
        return False
        
    return True

if __name__ == "__main__":
    test_lattice_application()