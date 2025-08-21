#!/usr/bin/env python3
"""Test runner for dynamic sequence meta-optimizer system.

This script provides an organized way to run the comprehensive test suite
for the meta-sequence optimizer system with different configurations and
reporting options.

Usage:
    python tests/run_dynamic_sequence_tests.py [options]
    
Options:
    --quick     Run only fast tests (exclude performance tests)
    --full      Run complete test suite including performance tests
    --coverage  Run with coverage reporting
    --verbose   Enable verbose output
    --parallel  Run tests in parallel where possible
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Optional


class DynamicSequenceTestRunner:
    """Test runner for dynamic sequence system."""
    
    def __init__(self):
        """Initialize the test runner."""
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent
        
        # Define test categories
        self.core_tests = [
            "test_dynamic_sequence_models.py",
            "test_dynamic_sequence_generation.py", 
            "test_sequence_engine_dynamic.py"
        ]
        
        self.integration_tests = [
            "test_backward_compatibility.py",
            "test_parallel_execution_integration.py"
        ]
        
        self.robustness_tests = [
            "test_error_handling_edge_cases.py"
        ]
        
        self.performance_tests = [
            "test_performance_memory.py"
        ]
        
        self.all_tests = (
            self.core_tests + 
            self.integration_tests + 
            self.robustness_tests + 
            self.performance_tests
        )
    
    def run_tests(
        self, 
        test_files: List[str],
        coverage: bool = False,
        verbose: bool = False,
        parallel: bool = False,
        extra_args: Optional[List[str]] = None
    ) -> int:
        """Run specified test files with given options.
        
        Args:
            test_files: List of test file names to run
            coverage: Whether to collect coverage data
            verbose: Whether to use verbose output
            parallel: Whether to run tests in parallel
            extra_args: Additional pytest arguments
            
        Returns:
            Exit code from pytest
        """
        cmd = ["python", "-m", "pytest"]
        
        # Add test files
        test_paths = [str(self.test_dir / test_file) for test_file in test_files]
        cmd.extend(test_paths)
        
        # Add pytest options
        if verbose:
            cmd.append("-v")
        
        if parallel:
            cmd.extend(["-n", "auto"])  # Requires pytest-xdist
        
        if coverage:
            cmd.extend([
                "--cov=open_deep_research.sequencing",
                "--cov-report=html",
                "--cov-report=term-missing"
            ])
        
        # Add extra arguments
        if extra_args:
            cmd.extend(extra_args)
        
        print(f"Running command: {' '.join(cmd)}")
        print("-" * 80)
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root)
        end_time = time.time()
        
        print("-" * 80)
        print(f"Tests completed in {end_time - start_time:.2f} seconds")
        
        return result.returncode
    
    def run_quick_tests(self, **kwargs) -> int:
        """Run quick tests (core + integration, no performance tests)."""
        print("🚀 Running Quick Test Suite")
        print("Running core functionality and integration tests...")
        
        quick_tests = self.core_tests + self.integration_tests + self.robustness_tests
        return self.run_tests(quick_tests, **kwargs)
    
    def run_full_tests(self, **kwargs) -> int:
        """Run complete test suite including performance tests."""
        print("🔬 Running Full Test Suite")
        print("Running all tests including performance and memory tests...")
        
        return self.run_tests(self.all_tests, **kwargs)
    
    def run_core_tests(self, **kwargs) -> int:
        """Run only core functionality tests."""
        print("⚡ Running Core Tests")
        print("Running core model, generation, and engine tests...")
        
        return self.run_tests(self.core_tests, **kwargs)
    
    def run_integration_tests(self, **kwargs) -> int:
        """Run only integration tests."""
        print("🔗 Running Integration Tests")
        print("Running backward compatibility and parallel execution tests...")
        
        return self.run_tests(self.integration_tests, **kwargs)
    
    def run_performance_tests(self, **kwargs) -> int:
        """Run only performance and memory tests."""
        print("📊 Running Performance Tests")
        print("Running performance and memory usage tests...")
        print("⚠️  These tests may take longer to complete...")
        
        return self.run_tests(self.performance_tests, **kwargs)
    
    def run_robustness_tests(self, **kwargs) -> int:
        """Run only error handling and edge case tests."""
        print("🛡️ Running Robustness Tests")
        print("Running error handling and edge case tests...")
        
        return self.run_tests(self.robustness_tests, **kwargs)
    
    def check_dependencies(self) -> bool:
        """Check if all required test dependencies are available."""
        print("🔍 Checking test dependencies...")
        
        required_packages = [
            "pytest",
            "pytest-asyncio", 
            "psutil"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package}")
                missing_packages.append(package)
        
        if missing_packages:
            print("\n⚠️  Missing required packages. Install with:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        
        print("✅ All dependencies satisfied")
        return True
    
    def show_test_summary(self):
        """Show summary of available tests."""
        print("📋 Dynamic Sequence Test Suite Summary")
        print("=" * 60)
        
        print(f"\n🔧 Core Tests ({len(self.core_tests)} files):")
        for test in self.core_tests:
            print(f"  • {test}")
        
        print(f"\n🔗 Integration Tests ({len(self.integration_tests)} files):")
        for test in self.integration_tests:
            print(f"  • {test}")
        
        print(f"\n🛡️ Robustness Tests ({len(self.robustness_tests)} files):")
        for test in self.robustness_tests:
            print(f"  • {test}")
        
        print(f"\n📊 Performance Tests ({len(self.performance_tests)} files):")
        for test in self.performance_tests:
            print(f"  • {test}")
        
        print(f"\n📈 Total: {len(self.all_tests)} test files")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run dynamic sequence meta-optimizer tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_dynamic_sequence_tests.py --quick
  python tests/run_dynamic_sequence_tests.py --full --coverage --verbose
  python tests/run_dynamic_sequence_tests.py --core
  python tests/run_dynamic_sequence_tests.py --performance
        """
    )
    
    # Test selection options
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--quick", action="store_true",
                           help="Run quick tests (core + integration, no performance)")
    test_group.add_argument("--full", action="store_true",
                           help="Run complete test suite including performance tests")
    test_group.add_argument("--core", action="store_true",
                           help="Run only core functionality tests")
    test_group.add_argument("--integration", action="store_true",
                           help="Run only integration tests")
    test_group.add_argument("--performance", action="store_true",
                           help="Run only performance tests")
    test_group.add_argument("--robustness", action="store_true",
                           help="Run only error handling tests")
    
    # Test execution options
    parser.add_argument("--coverage", action="store_true",
                       help="Collect coverage data")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--parallel", action="store_true",
                       help="Run tests in parallel (requires pytest-xdist)")
    parser.add_argument("--check-deps", action="store_true",
                       help="Check test dependencies and exit")
    parser.add_argument("--summary", action="store_true",
                       help="Show test summary and exit")
    
    args = parser.parse_args()
    
    runner = DynamicSequenceTestRunner()
    
    # Handle special options
    if args.check_deps:
        success = runner.check_dependencies()
        sys.exit(0 if success else 1)
    
    if args.summary:
        runner.show_test_summary()
        sys.exit(0)
    
    # Check dependencies before running tests
    if not runner.check_dependencies():
        sys.exit(1)
    
    # Prepare test options
    test_options = {
        "coverage": args.coverage,
        "verbose": args.verbose,
        "parallel": args.parallel
    }
    
    # Run selected test suite
    exit_code = 0
    
    try:
        if args.quick:
            exit_code = runner.run_quick_tests(**test_options)
        elif args.full:
            exit_code = runner.run_full_tests(**test_options)
        elif args.core:
            exit_code = runner.run_core_tests(**test_options)
        elif args.integration:
            exit_code = runner.run_integration_tests(**test_options)
        elif args.performance:
            exit_code = runner.run_performance_tests(**test_options)
        elif args.robustness:
            exit_code = runner.run_robustness_tests(**test_options)
        else:
            # Default to quick tests
            print("No specific test suite selected, running quick tests...")
            exit_code = runner.run_quick_tests(**test_options)
    
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        exit_code = 130
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        exit_code = 1
    
    # Summary
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()