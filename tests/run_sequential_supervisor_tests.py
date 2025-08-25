"""Comprehensive test runner for Sequential Multi-Agent Supervisor Architecture.

This script executes all tests for the Sequential Supervisor system and generates
a comprehensive test execution report with pass/fail status and coverage metrics.

Test Execution Categories:
1. Sequential Supervisor Integration Tests
2. Agent Registry Loading Tests
3. Completion Detection Tests
4. Configuration System Tests
5. Sequence Generation and LLM Judge Tests
6. Backward Compatibility Tests
7. Performance Benchmark Tests

Features:
- Parallel test execution where safe
- Comprehensive coverage reporting
- Performance benchmark integration
- HTML and JSON report generation
- CI/CD integration support
- Failure analysis and debugging info
"""

import sys
import os
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
import concurrent.futures
from enum import Enum


class TestStatus(Enum):
    """Test execution status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    NOT_RUN = "not_run"


@dataclass
class TestResult:
    """Individual test result container."""
    test_file: str
    test_name: str
    status: TestStatus
    duration: float
    error_message: Optional[str] = None
    coverage_percentage: Optional[float] = None
    performance_metrics: Optional[Dict[str, Any]] = None


@dataclass
class TestSuiteResult:
    """Complete test suite result container."""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    total_duration: float
    coverage_percentage: float
    test_results: List[TestResult]
    timestamp: str
    requirements_met: bool = True


class SequentialSupervisorTestRunner:
    """Comprehensive test runner for Sequential Supervisor system."""
    
    def __init__(self, verbose: bool = False, parallel: bool = True, coverage: bool = True):
        self.verbose = verbose
        self.parallel = parallel
        self.coverage = coverage
        self.results: List[TestSuiteResult] = []
        self.start_time = time.time()
        
        # Test configuration
        self.test_files = [
            "test_sequential_supervisor_integration.py",
            "test_agent_registry_loading.py", 
            "test_completion_detection.py",
            "test_configuration_system.py",
            "test_sequence_generation_llm_judge.py",
            "test_comprehensive_backward_compatibility.py",
            "test_performance_benchmarks.py"
        ]
        
        # Performance requirements
        self.performance_requirements = {
            "handoff_timing_seconds": 3.0,
            "agent_loading_seconds": 2.0,
            "completion_detection_seconds": 1.0,
            "memory_increase_mb": 100.0,
            "test_success_rate": 0.95
        }
        
        # Coverage requirements
        self.coverage_requirements = {
            "minimum_total_coverage": 95.0,
            "minimum_module_coverage": 85.0
        }
    
    def check_dependencies(self) -> bool:
        """Check that all required dependencies are available."""
        required_packages = ["pytest", "pytest-asyncio", "pytest-cov", "psutil"]
        
        print("🔍 Checking test dependencies...")
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"  ✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"  ❌ {package}")
        
        if missing_packages:
            print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        print("✅ All dependencies available")
        return True
    
    def run_test_file(self, test_file: str, timeout: int = 300) -> TestSuiteResult:
        """Run a single test file and capture results."""
        print(f"🧪 Running {test_file}...")
        start_time = time.time()
        
        # Construct pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            test_file,
            "-v",
            "--tb=short",
            "--json-report",
            f"--json-report-file={test_file}.json"
        ]
        
        if self.coverage:
            cmd.extend([
                "--cov=open_deep_research",
                f"--cov-report=json:{test_file}_coverage.json",
                "--cov-report=term-missing"
            ])
        
        try:
            # Run pytest
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            # Parse JSON report if available
            json_report_path = Path(__file__).parent / f"{test_file}.json"
            test_results = []
            
            if json_report_path.exists():
                with open(json_report_path) as f:
                    json_data = json.load(f)
                    
                for test in json_data.get("tests", []):
                    test_result = TestResult(
                        test_file=test_file,
                        test_name=test["nodeid"],
                        status=TestStatus(test["outcome"]),
                        duration=test.get("duration", 0.0),
                        error_message=test.get("call", {}).get("longrepr") if test["outcome"] != "passed" else None
                    )
                    test_results.append(test_result)
                
                # Clean up JSON report
                json_report_path.unlink()
            
            # Parse coverage if available
            coverage_percentage = 0.0
            coverage_path = Path(__file__).parent / f"{test_file}_coverage.json"
            if coverage_path.exists():
                with open(coverage_path) as f:
                    coverage_data = json.load(f)
                    coverage_percentage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
                coverage_path.unlink()
            
            # Count results
            passed = len([r for r in test_results if r.status == TestStatus.PASSED])
            failed = len([r for r in test_results if r.status == TestStatus.FAILED])
            skipped = len([r for r in test_results if r.status == TestStatus.SKIPPED])
            errors = len([r for r in test_results if r.status == TestStatus.ERROR])
            
            suite_result = TestSuiteResult(
                suite_name=test_file,
                total_tests=len(test_results),
                passed=passed,
                failed=failed,
                skipped=skipped,
                errors=errors,
                total_duration=duration,
                coverage_percentage=coverage_percentage,
                test_results=test_results,
                timestamp=datetime.now().isoformat(),
                requirements_met=(failed + errors) == 0 and coverage_percentage >= self.coverage_requirements["minimum_module_coverage"]
            )
            
            status_symbol = "✅" if suite_result.requirements_met else "❌"
            print(f"  {status_symbol} {test_file}: {passed}P {failed}F {skipped}S {errors}E ({duration:.1f}s, {coverage_percentage:.1f}% cov)")
            
            return suite_result
            
        except subprocess.TimeoutExpired:
            print(f"  ⏱️  {test_file}: TIMEOUT after {timeout}s")
            return TestSuiteResult(
                suite_name=test_file,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                total_duration=timeout,
                coverage_percentage=0.0,
                test_results=[
                    TestResult(
                        test_file=test_file,
                        test_name=f"{test_file}::timeout",
                        status=TestStatus.ERROR,
                        duration=timeout,
                        error_message=f"Test execution timed out after {timeout} seconds"
                    )
                ],
                timestamp=datetime.now().isoformat(),
                requirements_met=False
            )
        
        except Exception as e:
            print(f"  💥 {test_file}: ERROR - {e}")
            return TestSuiteResult(
                suite_name=test_file,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                total_duration=time.time() - start_time,
                coverage_percentage=0.0,
                test_results=[
                    TestResult(
                        test_file=test_file,
                        test_name=f"{test_file}::exception",
                        status=TestStatus.ERROR,
                        duration=0.0,
                        error_message=str(e)
                    )
                ],
                timestamp=datetime.now().isoformat(),
                requirements_met=False
            )
    
    def run_all_tests(self) -> bool:
        """Run all test suites and collect results."""
        print("🚀 Starting Sequential Multi-Agent Supervisor Test Suite")
        print(f"📊 Running {len(self.test_files)} test files...")
        print(f"⚙️  Parallel: {self.parallel}, Coverage: {self.coverage}")
        print("")
        
        if self.parallel:
            # Run tests in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_test = {
                    executor.submit(self.run_test_file, test_file): test_file 
                    for test_file in self.test_files
                }
                
                for future in concurrent.futures.as_completed(future_to_test):
                    test_file = future_to_test[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                    except Exception as e:
                        print(f"💥 Exception running {test_file}: {e}")
        else:
            # Run tests sequentially
            for test_file in self.test_files:
                result = self.run_test_file(test_file)
                self.results.append(result)
        
        print("")
        return self.analyze_results()
    
    def analyze_results(self) -> bool:
        """Analyze test results and check requirements."""
        total_duration = time.time() - self.start_time
        
        # Calculate totals
        total_tests = sum(r.total_tests for r in self.results)
        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        total_errors = sum(r.errors for r in self.results)
        
        # Calculate coverage
        if self.results:
            avg_coverage = sum(r.coverage_percentage for r in self.results) / len(self.results)
        else:
            avg_coverage = 0.0
        
        # Print summary
        print("📋 Test Execution Summary")
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed} ✅")
        print(f"Failed: {total_failed} ❌")
        print(f"Skipped: {total_skipped} ⏭️")
        print(f"Errors: {total_errors} 💥")
        print(f"Success Rate: {(total_passed/total_tests*100) if total_tests > 0 else 0:.1f}%")
        print(f"Coverage: {avg_coverage:.1f}%")
        print(f"Total Duration: {total_duration:.1f}s")
        print("")
        
        # Check requirements
        requirements_met = True
        success_rate = (total_passed / total_tests) if total_tests > 0 else 0.0
        
        print("✅ Requirements Analysis")
        print("-" * 30)
        
        # Success rate requirement
        if success_rate >= self.performance_requirements["test_success_rate"]:
            print(f"✅ Test Success Rate: {success_rate:.1%} >= {self.performance_requirements['test_success_rate']:.1%}")
        else:
            print(f"❌ Test Success Rate: {success_rate:.1%} < {self.performance_requirements['test_success_rate']:.1%}")
            requirements_met = False
        
        # Coverage requirement
        if avg_coverage >= self.coverage_requirements["minimum_total_coverage"]:
            print(f"✅ Code Coverage: {avg_coverage:.1f}% >= {self.coverage_requirements['minimum_total_coverage']:.1f}%")
        else:
            print(f"❌ Code Coverage: {avg_coverage:.1f}% < {self.coverage_requirements['minimum_total_coverage']:.1f}%")
            requirements_met = False
        
        # Individual suite requirements
        failing_suites = [r for r in self.results if not r.requirements_met]
        if not failing_suites:
            print("✅ All test suites meet individual requirements")
        else:
            print(f"❌ {len(failing_suites)} test suites fail requirements:")
            for suite in failing_suites:
                print(f"   - {suite.suite_name}")
            requirements_met = False
        
        print("")
        
        if requirements_met:
            print("🎉 All requirements met! Sequential Supervisor system is production-ready.")
        else:
            print("⚠️  Some requirements not met. Review failures before deployment.")
        
        print("")
        return requirements_met
    
    def generate_html_report(self, output_path: str = "test_report.html") -> str:
        """Generate comprehensive HTML test report."""
        html_template = '''<!DOCTYPE html>
<html>
<head>
    <title>Sequential Multi-Agent Supervisor Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .summary { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .test-suite { background: white; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .suite-header { background: #34495e; color: white; padding: 15px; border-radius: 8px 8px 0 0; }
        .suite-body { padding: 20px; }
        .test-result { padding: 10px; border-bottom: 1px solid #eee; }
        .test-result:last-child { border-bottom: none; }
        .passed { color: #27ae60; }
        .failed { color: #e74c3c; }
        .skipped { color: #f39c12; }
        .error { color: #c0392b; }
        .metric { display: inline-block; margin-right: 20px; padding: 10px; background: #ecf0f1; border-radius: 4px; }
        .requirements { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .requirement-met { color: #27ae60; }
        .requirement-failed { color: #e74c3c; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Sequential Multi-Agent Supervisor Test Report</h1>
        <p>Generated: {timestamp}</p>
    </div>
    
    <div class="summary">
        <h2>📊 Test Execution Summary</h2>
        <div class="metric">Total Tests: {total_tests}</div>
        <div class="metric">Passed: {total_passed}</div>
        <div class="metric">Failed: {total_failed}</div>
        <div class="metric">Errors: {total_errors}</div>
        <div class="metric">Success Rate: {success_rate:.1%}</div>
        <div class="metric">Coverage: {avg_coverage:.1f}%</div>
        <div class="metric">Duration: {total_duration:.1f}s</div>
    </div>
    
    <div class="requirements">
        <h2>✅ Requirements Analysis</h2>
        {requirements_html}
    </div>
    
    <h2>🧪 Test Suite Details</h2>
    {test_suites_html}
</body>
</html>'''
        
        # Calculate summary metrics
        total_tests = sum(r.total_tests for r in self.results)
        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_errors = sum(r.errors for r in self.results)
        success_rate = (total_passed / total_tests) if total_tests > 0 else 0.0
        avg_coverage = sum(r.coverage_percentage for r in self.results) / len(self.results) if self.results else 0.0
        total_duration = time.time() - self.start_time
        
        # Generate requirements HTML
        requirements_html = []
        if success_rate >= self.performance_requirements["test_success_rate"]:
            requirements_html.append(f'<div class="requirement-met">✅ Test Success Rate: {success_rate:.1%}</div>')
        else:
            requirements_html.append(f'<div class="requirement-failed">❌ Test Success Rate: {success_rate:.1%}</div>')
        
        if avg_coverage >= self.coverage_requirements["minimum_total_coverage"]:
            requirements_html.append(f'<div class="requirement-met">✅ Code Coverage: {avg_coverage:.1f}%</div>')
        else:
            requirements_html.append(f'<div class="requirement-failed">❌ Code Coverage: {avg_coverage:.1f}%</div>')
        
        # Generate test suites HTML
        test_suites_html = []
        for suite in self.results:
            suite_status = "✅" if suite.requirements_met else "❌"
            test_results_html = []
            
            for test in suite.test_results:
                status_class = test.status.value
                test_results_html.append(f'''
                    <div class="test-result">
                        <span class="{status_class}">{test.status.value.upper()}</span>
                        <strong>{test.test_name}</strong>
                        <span>({test.duration:.2f}s)</span>
                        {f"<br><small>{test.error_message}</small>" if test.error_message else ""}
                    </div>
                ''')
            
            test_suites_html.append(f'''
                <div class="test-suite">
                    <div class="suite-header">
                        <h3>{suite_status} {suite.suite_name}</h3>
                        <div>
                            {suite.passed}P {suite.failed}F {suite.skipped}S {suite.errors}E 
                            ({suite.total_duration:.1f}s, {suite.coverage_percentage:.1f}% coverage)
                        </div>
                    </div>
                    <div class="suite-body">
                        {''.join(test_results_html)}
                    </div>
                </div>
            ''')
        
        # Generate final HTML
        html_content = html_template.format(
            timestamp=datetime.now().isoformat(),
            total_tests=total_tests,
            total_passed=total_passed,
            total_failed=total_failed,
            total_errors=total_errors,
            success_rate=success_rate,
            avg_coverage=avg_coverage,
            total_duration=total_duration,
            requirements_html=''.join(requirements_html),
            test_suites_html=''.join(test_suites_html)
        )
        
        # Write HTML report
        Path(output_path).write_text(html_content)
        print(f"📄 HTML report generated: {output_path}")
        return output_path
    
    def generate_json_report(self, output_path: str = "test_report.json") -> str:
        """Generate JSON test report for CI/CD integration."""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": time.time() - self.start_time,
            "summary": {
                "total_tests": sum(r.total_tests for r in self.results),
                "total_passed": sum(r.passed for r in self.results),
                "total_failed": sum(r.failed for r in self.results),
                "total_skipped": sum(r.skipped for r in self.results),
                "total_errors": sum(r.errors for r in self.results),
                "success_rate": (sum(r.passed for r in self.results) / sum(r.total_tests for r in self.results)) if sum(r.total_tests for r in self.results) > 0 else 0.0,
                "average_coverage": sum(r.coverage_percentage for r in self.results) / len(self.results) if self.results else 0.0
            },
            "requirements": {
                "performance_requirements": self.performance_requirements,
                "coverage_requirements": self.coverage_requirements,
                "all_requirements_met": all(r.requirements_met for r in self.results)
            },
            "test_suites": [asdict(result) for result in self.results]
        }
        
        # Write JSON report
        Path(output_path).write_text(json.dumps(report_data, indent=2))
        print(f"📊 JSON report generated: {output_path}")
        return output_path


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="Sequential Multi-Agent Supervisor Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel test execution")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--output-dir", default=".", help="Output directory for reports")
    parser.add_argument("--timeout", type=int, default=300, help="Test timeout in seconds")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = SequentialSupervisorTestRunner(
        verbose=args.verbose,
        parallel=not args.no_parallel,
        coverage=not args.no_coverage
    )
    
    # Check dependencies
    if not runner.check_dependencies():
        sys.exit(1)
    
    print("")
    
    # Run all tests
    success = runner.run_all_tests()
    
    # Generate reports
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    html_report = output_dir / "sequential_supervisor_test_report.html"
    json_report = output_dir / "sequential_supervisor_test_report.json"
    
    runner.generate_html_report(str(html_report))
    runner.generate_json_report(str(json_report))
    
    print(f"📁 Reports saved to: {output_dir}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()