"""
Measure a single SWE-Perf instance with all metrics.
Optimized for compatibility with legacy scientific Python projects.
Supports multiple Python versions based on repository requirements.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import subprocess
import tempfile
import shutil
import os
from typing import Dict, Optional, List, Tuple
from src.utils.config import load_config
from src.measurement.collector import MetricsCollector

# Default Python version
DEFAULT_PYTHON = "python3.9"

# Conda path for Python 3.6 environments
CONDA_PATH = "/opt/miniconda3/bin/conda"

# Python version mapping based on SWE-Perf constants.py
# Format: (repo_lower, version) -> python_executable
PYTHON_VERSION_MAP = {
    # xarray requires Python 3.10
    ('pydata/xarray', '0.18'): 'python3.10',
    ('pydata/xarray', '0.19'): 'python3.10',
    ('pydata/xarray', '0.2'): 'python3.10',
    ('pydata/xarray', '0.20'): 'python3.10',
    ('pydata/xarray', '2022.03'): 'python3.10',
    ('pydata/xarray', '2022.06'): 'python3.10',
    ('pydata/xarray', '2022.09'): 'python3.10',
    ('pydata/xarray', '2023.04'): 'python3.10',
    ('pydata/xarray', '2023.07'): 'python3.10',
    ('pydata/xarray', '2024.05'): 'python3.10',
    # astropy v5.3 requires Python 3.10
    ('astropy/astropy', 'v5.3'): 'python3.10',
}

# Sklearn versions that require conda with Python 3.6
SKLEARN_CONDA_VERSIONS = ['0.2', '0.20', '0.21', '0.22']

# Astropy versions that require conda with Python 3.10
ASTROPY_CONDA_VERSIONS = ['v5.3']

# Repository-specific package constraints
REPO_PACKAGE_CONSTRAINTS = {
    'scikit-learn/scikit-learn': {
        'numpy': '<1.24',
        'cython': '<3.0',
        'setuptools': '<70',
        'scipy': '>=1.0,<1.14',
    },
    'pydata/xarray': {
        'numpy': '<2.0',
        'setuptools': '<70',
        'pandas': '<2.1',
    },
    'astropy/astropy': {
        'numpy': '==1.25.2',
        'cython': '<3.0',
        'setuptools': '==68.0.0',
    },
    'default': {
        'setuptools': '<70',
        'numpy': '<2.0',
        'matplotlib': '<3.9',
        'cython': '<3.0',
    }
}

# Pre-install commands for specific repos
REPO_PRE_INSTALL = {
    'astropy/astropy': [
        'sed -i \'s/requires = \\["setuptools",/requires = \\["setuptools==68.0.0",/\' pyproject.toml'
    ],
}

# Conda-specific constraints for sklearn with Python 3.6
SKLEARN_CONDA_CONSTRAINTS = {
    'numpy': '1.19.2',
    'cython': '0.29.24',
    'setuptools': '58.0.4',
    'scipy': '1.5.2',
}

# Conda-specific constraints for astropy with Python 3.10
ASTROPY_CONDA_CONSTRAINTS = {
    'numpy': '1.25.2',
    'cython': '0.29.37',
    'setuptools': '68.0.0',
}

# Special installation procedures for specific repos
REPO_SPECIAL_INSTALL = {
    'scikit-learn/scikit-learn': 'sklearn_install',
}


def should_use_conda(repo: str, version: str) -> bool:
    """Check if we should use conda for this repo/version."""
    repo_lower = repo.lower()
    if 'scikit-learn' in repo_lower and version in SKLEARN_CONDA_VERSIONS:
        if Path(CONDA_PATH).exists():
            return True
    if 'astropy' in repo_lower and version in ASTROPY_CONDA_VERSIONS:
        if Path(CONDA_PATH).exists():
            return True
    return False


def get_python_executable(repo: str, version: str) -> str:
    """Get the appropriate Python executable for a repo/version."""
    repo_lower = repo.lower()
    
    key = (repo_lower, version)
    if key in PYTHON_VERSION_MAP:
        py_exec = PYTHON_VERSION_MAP[key]
        try:
            subprocess.run([py_exec, '--version'], capture_output=True, check=True)
            return py_exec
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"  ⚠️ {py_exec} not available, falling back to {DEFAULT_PYTHON}")
            return DEFAULT_PYTHON
    
    for (r, v), py in PYTHON_VERSION_MAP.items():
        if r == repo_lower and v == '*':
            try:
                subprocess.run([py, '--version'], capture_output=True, check=True)
                return py
            except:
                pass
    
    return DEFAULT_PYTHON


def get_package_constraints(repo: str) -> Dict[str, str]:
    """Get package version constraints for a repository."""
    repo_lower = repo.lower()
    
    if repo_lower in REPO_PACKAGE_CONSTRAINTS:
        return REPO_PACKAGE_CONSTRAINTS[repo_lower]
    
    return REPO_PACKAGE_CONSTRAINTS['default']


def run_pre_install_commands(repo: str, repo_path: Path) -> bool:
    """Run pre-install commands for a repository."""
    repo_lower = repo.lower()
    
    if repo_lower not in REPO_PRE_INSTALL:
        return True
    
    print(f"  🔧 Running pre-install commands for {repo}...")
    
    for cmd in REPO_PRE_INSTALL[repo_lower]:
        try:
            subprocess.run(
                cmd,
                shell=True,
                cwd=repo_path,
                check=True,
                timeout=60
            )
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️ Pre-install command failed: {cmd}")
        except subprocess.TimeoutExpired:
            print(f"  ⚠️ Pre-install command timed out: {cmd}")
    
    return True


def get_conda_env_python(env_name: str) -> Optional[str]:
    """
    Get the python path for a conda environment.
    Tries multiple possible locations.
    """
    possible_paths = [
        Path.home() / '.conda' / 'envs' / env_name / 'bin' / 'python',
        Path('/opt/miniconda3/envs') / env_name / 'bin' / 'python',
        Path.home() / 'miniconda3' / 'envs' / env_name / 'bin' / 'python',
        Path.home() / 'anaconda3' / 'envs' / env_name / 'bin' / 'python',
    ]
    
    for p in possible_paths:
        if p.exists():
            return str(p)
    
    # Try using conda to find it
    try:
        result = subprocess.run(
            [CONDA_PATH, 'env', 'list'],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if env_name in line:
                parts = line.split()
                for part in parts:
                    if '/' in part and env_name in part:
                        python_path = Path(part) / 'bin' / 'python'
                        if python_path.exists():
                            return str(python_path)
    except:
        pass
    
    return None


class SWEPerfMeasurer:
    """Measure SWE-Perf instance with green metrics."""
    
    def __init__(self, dataset_path: str, country_code: Optional[str] = None):
        """Initialize measurer."""
        self.dataset_path = Path(dataset_path)
        self.country_code = country_code
        self.config = load_config()
        
        # Verify default Python is available
        try:
            result = subprocess.run(
                [DEFAULT_PYTHON, '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ Default Python: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                f"❌ {DEFAULT_PYTHON} not found! "
                "SWE-Perf requires Python 3.9 for compatibility."
            )
        
        # Check for conda
        if Path(CONDA_PATH).exists():
            result = subprocess.run(
                [CONDA_PATH, '--version'],
                capture_output=True,
                text=True
            )
            print(f"✅ Conda: {result.stdout.strip()}")
        else:
            print("⚠️  Conda not found - sklearn 0.21 with Python 3.6 won't work")
        
        # Check for additional Python versions
        for py_version in ['python3.10', 'python3.11', 'python3.6']:
            try:
                result = subprocess.run(
                    [py_version, '--version'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"✅ Available: {result.stdout.strip()}")
            except:
                pass
        
        # Load dataset
        print(f"📂 Loading dataset from {self.dataset_path}...")
        with open(self.dataset_path, 'r') as f:
            self.dataset = json.load(f)
        print(f"✅ Loaded {len(self.dataset)} instances")
    
    def get_instance(self, instance_id: str) -> Optional[Dict]:
        """Get instance by ID."""
        for instance in self.dataset:
            if instance['instance_id'] == instance_id:
                return instance
        return None
    
    def setup_repository(self, instance: Dict, temp_dir: Path, commit: str) -> Path:
        """Clone repository and checkout specific commit."""
        repo_name = instance['repo']
        repo_url = f"https://github.com/{repo_name}.git"
        
        repo_path = temp_dir / f"{repo_name.split('/')[-1]}_{commit[:8]}"
        
        print(f"  📦 Cloning {repo_name}...")
        
        subprocess.run(
            ['git', 'clone', repo_url, str(repo_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        
        print(f"  🔀 Fetching commit {commit[:8]}...")
        subprocess.run(
            ['git', 'fetch', 'origin', commit],
            cwd=repo_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
        
        print(f"  🔀 Checking out commit {commit[:8]}...")
        subprocess.run(
            ['git', 'checkout', commit],
            cwd=repo_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        
        return repo_path
    
    def install_sklearn_conda(self, repo_path: Path, env_name: str) -> Optional[str]:
        """Install sklearn using conda with Python 3.6."""
        try:
            print(f"  📦 [conda] Creating environment {env_name} with Python 3.6...")
            subprocess.run(
                [CONDA_PATH, 'create', '-n', env_name, 'python=3.6', '-y'],
                check=True,
                timeout=300,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Find the conda python
            conda_python = get_conda_env_python(env_name)
            if not conda_python:
                raise RuntimeError(f"Could not find python for conda env {env_name}")
            
            conda_pip = str(Path(conda_python).parent / 'pip')
            
            print(f"  📦 [conda] Installing build dependencies...")
            subprocess.run(
                [conda_pip, 'install',
                 f"numpy=={SKLEARN_CONDA_CONSTRAINTS['numpy']}",
                 f"scipy=={SKLEARN_CONDA_CONSTRAINTS['scipy']}",
                 f"cython=={SKLEARN_CONDA_CONSTRAINTS['cython']}",
                 f"setuptools=={SKLEARN_CONDA_CONSTRAINTS['setuptools']}",
                 'pytest', 'joblib'],
                check=True,
                timeout=300
            )
            
            print(f"  📦 [conda] Building sklearn...")
            subprocess.run(
                [conda_pip, 'install', '-e', '.', '--no-build-isolation'],
                cwd=repo_path,
                check=True,
                timeout=600
            )
            
            # Verify python still exists
            if not Path(conda_python).exists():
                raise RuntimeError(f"Python disappeared: {conda_python}")
            
            print(f"  ✅ [conda] sklearn installation complete")
            print(f"  📍 Python path: {conda_python}")
            return conda_python
            
        except Exception as e:
            print(f"  ❌ [conda] Installation failed: {e}")
            subprocess.run(
                [CONDA_PATH, 'env', 'remove', '-n', env_name, '-y'],
                capture_output=True
            )
            return None
    
    def install_astropy_conda(self, repo_path: Path, env_name: str) -> Optional[str]:
        """Install astropy using conda with Python 3.10."""
        try:
            print(f"  📦 [conda] Creating environment {env_name} with Python 3.10...")
            subprocess.run(
                [CONDA_PATH, 'create', '-n', env_name, 'python=3.10', '-y'],
                check=True,
                timeout=300,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Find the conda python
            conda_python = get_conda_env_python(env_name)
            if not conda_python:
                raise RuntimeError(f"Could not find python for conda env {env_name}")
            
            conda_pip = str(Path(conda_python).parent / 'pip')
            
            print(f"  📦 [conda] Installing base packages...")
            subprocess.run(
                [conda_pip, 'install', '--upgrade', 'pip', 
                 f"setuptools=={ASTROPY_CONDA_CONSTRAINTS['setuptools']}", 'wheel'],
                check=True,
                timeout=120
            )
            
            run_pre_install_commands('astropy/astropy', repo_path)
            
            print(f"  📦 [conda] Installing astropy with build isolation...")
            subprocess.run(
                [conda_pip, 'install', '-e', '.[test]', '--verbose'],
                cwd=repo_path,
                check=True,
                timeout=900
            )
            
            # Verify python still exists
            if not Path(conda_python).exists():
                raise RuntimeError(f"Python disappeared: {conda_python}")
            
            print(f"  ✅ [conda] astropy installation complete")
            print(f"  📍 Python path: {conda_python}")
            return conda_python
            
        except Exception as e:
            print(f"  ❌ [conda] Installation failed: {e}")
            subprocess.run(
                [CONDA_PATH, 'env', 'remove', '-n', env_name, '-y'],
                capture_output=True
            )
            return None
    
    def cleanup_conda_env(self, env_name: str):
        """Remove a conda environment."""
        if env_name:
            print(f"  🧹 Cleaning up conda env: {env_name}")
            subprocess.run(
                [CONDA_PATH, 'env', 'remove', '-n', env_name, '-y'],
                capture_output=True
            )
    
    def install_sklearn(self, repo_path: Path, venv_path: Path, constraints: Dict[str, str]) -> bool:
        """Special installation procedure for scikit-learn 0.21."""
        venv_pip = str(venv_path / 'bin' / 'pip')
        venv_python = str(venv_path / 'bin' / 'python')
        
        try:
            print(f"  📦 [sklearn] Installing build dependencies...")
            numpy_constraint = constraints.get('numpy', '<1.24')
            cython_constraint = constraints.get('cython', '<3.0')
            scipy_constraint = constraints.get('scipy', '>=1.0,<1.14')
            
            subprocess.run(
                [venv_pip, 'install',
                 f'numpy{numpy_constraint}',
                 f'cython{cython_constraint}',
                 f'scipy{scipy_constraint}',
                 'joblib', 'pytest'],
                check=True,
                timeout=300
            )
            
            print(f"  📦 [sklearn] Building sklearn...")
            subprocess.run(
                [venv_pip, 'install', '-e', '.', '--no-build-isolation'],
                cwd=repo_path,
                check=True,
                timeout=600
            )
            
            print(f"  🔧 [sklearn] Fixing joblib compatibility...")
            
            externals_joblib = repo_path / 'sklearn' / 'externals' / 'joblib'
            if externals_joblib.exists():
                shutil.rmtree(externals_joblib)
                externals_joblib.mkdir(parents=True)
                init_file = externals_joblib / '__init__.py'
                init_file.write_text('''# Stub to redirect to external joblib (fixes Python 3.9 compatibility)
from joblib import *
from joblib import Parallel, delayed, Memory, parallel_backend
from joblib import register_parallel_backend, cpu_count, effective_n_jobs
from joblib import hash, dump, load, __version__
try:
    from joblib import parallel
except ImportError:
    pass
import logging
logger = logging.getLogger(__name__)
''')
            
            print(f"  ✅ [sklearn] Installation complete")
            return True
            
        except Exception as e:
            print(f"  ❌ [sklearn] Installation failed: {e}")
            return False
    
    def install_dependencies(self, repo_path: Path, repo: str, version: str, commit: str) -> Tuple[Optional[str], Optional[str]]:
        """Create virtual environment and install package dependencies."""
        repo_lower = repo.lower()
        
        # Check if we should use conda for sklearn
        if 'scikit-learn' in repo_lower and version in SKLEARN_CONDA_VERSIONS:
            if Path(CONDA_PATH).exists():
                env_name = f"sklearn_{commit[:8]}"
                python_path = self.install_sklearn_conda(repo_path, env_name)
                if python_path:
                    return (python_path, env_name)
                else:
                    print(f"  ⚠️ Conda failed, falling back to venv...")
        
        # Check if we should use conda for astropy
        if 'astropy' in repo_lower and version in ASTROPY_CONDA_VERSIONS:
            if Path(CONDA_PATH).exists():
                env_name = f"astropy_{commit[:8]}"
                python_path = self.install_astropy_conda(repo_path, env_name)
                if python_path:
                    return (python_path, env_name)
                else:
                    print(f"  ⚠️ Conda failed, falling back to venv...")
        
        # Get appropriate Python version
        python_exec = get_python_executable(repo, version)
        constraints = get_package_constraints(repo)
        
        print(f"  📦 Creating virtual environment with {python_exec}...")
        
        venv_path = repo_path / "venv_sweperf"
        venv_pip = str(venv_path / 'bin' / 'pip')
        
        try:
            subprocess.run(
                [python_exec, '-m', 'venv', str(venv_path)],
                check=True,
                timeout=60
            )
            print(f"  ✅ Virtual environment created with {python_exec}")
            
            setuptools_constraint = constraints.get('setuptools', '<70')
            print(f"  📦 Installing base packages (setuptools{setuptools_constraint})...")
            subprocess.run(
                [venv_pip, 'install', '--upgrade', 
                 'pip', f"setuptools{setuptools_constraint}", 'wheel'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120
            )
            
            run_pre_install_commands(repo, repo_path)
            
            if repo_lower in REPO_SPECIAL_INSTALL:
                special_method = REPO_SPECIAL_INSTALL[repo_lower]
                if special_method == 'sklearn_install':
                    success = self.install_sklearn(repo_path, venv_path, constraints)
                    if success:
                        return (str(venv_path / 'bin' / 'python'), None)
                    else:
                        return (None, None)
            
            numpy_constraint = constraints.get('numpy', '<2.0')
            cython_constraint = constraints.get('cython', '<3.0')
            
            print(f"  📦 Installing build dependencies (numpy{numpy_constraint}, cython{cython_constraint})...")
            subprocess.run(
                [venv_pip, 'install',
                 'extension_helpers', 'setuptools_scm', 
                 f"cython{cython_constraint}", 
                 f"numpy{numpy_constraint}",
                 "scipy"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=300
            )
            
            print(f"  📦 Installing project dependencies (version: {version})...")
            subprocess.run(
                [venv_pip, 'install', '-e', '.', '--no-build-isolation'],
                cwd=repo_path,
                check=True,
                timeout=600
            )
            
            matplotlib_constraint = constraints.get('matplotlib', '<3.9')
            print(f"  📦 Installing test dependencies (matplotlib{matplotlib_constraint})...")
            subprocess.run(
                [venv_pip, 'install', 
                 'pytest', 'hypothesis', 'scipy', 'pytest-astropy', 'urllib3',
                 f"matplotlib{matplotlib_constraint}",
                 f"numpy{numpy_constraint}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=180
            )
            
            print(f"  ✅ Dependencies installed")
            return (str(venv_path / 'bin' / 'python'), None)
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  ⚠️  Warning: Could not install dependencies: {e}")
            return (None, None)
    
    def verify_python_path(self, python_path: str) -> bool:
        """Verify that a python path exists and works."""
        if not python_path:
            return False
        
        if not Path(python_path).exists():
            print(f"  ⚠️ Python path does not exist: {python_path}")
            return False
        
        try:
            result = subprocess.run(
                [python_path, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            print(f"  📍 Python verified: {result.stdout.strip()}")
            return True
        except Exception as e:
            print(f"  ⚠️ Python verification failed: {e}")
            return False
    
    def measure_single_test(
        self,
        collector: MetricsCollector,
        test_name: str,
        repo_path: Path,
        python_path: str,
        repetitions: int
    ) -> Optional[Dict]:
        """Measure a single test with error handling."""
        try:
            # Verify python exists before running
            if not Path(python_path).exists():
                raise RuntimeError(f"Python not found: {python_path}")
            
            # Build pytest command - use absolute paths
            repo_abs = str(repo_path.absolute())
            test_command = f"cd {repo_abs} && {python_path} -m pytest '{test_name}' -v"
            
            # Measure test execution
            test_results = collector.measure_test_execution(
                test_command=test_command,
                repetitions=repetitions
            )
            
            test_results['test_name'] = test_name
            test_results['status'] = 'success'
            return test_results
            
        except Exception as e:
            print(f"    ❌ Test failed: {str(e)[:100]}")
            return {
                'test_name': test_name,
                'status': 'failed',
                'error': str(e)[:500]
            }
    
    def measure_commit(
        self,
        instance: Dict,
        commit: str,
        commit_type: str,
        temp_dir: Path
    ) -> Tuple[Dict, Optional[str]]:
        """
        Measure metrics for a specific commit.
        Returns tuple of (results, conda_env_name) so cleanup can happen later.
        """
        print(f"\n🔬 Measuring {commit_type} commit...")
        print("=" * 60)
        
        # Setup repository
        repo_path = self.setup_repository(instance, temp_dir, commit)
        
        # Install dependencies with repo-specific handling
        python_path, conda_env = self.install_dependencies(
            repo_path, 
            instance['repo'], 
            instance['version'],
            commit
        )
        
        if python_path is None:
            print(f"  ⚠️  Skipping measurements - dependencies failed")
            return ({'status': 'dependency_failed'}, conda_env)
        
        # Verify python path
        if not self.verify_python_path(python_path):
            print(f"  ⚠️  Python path invalid - skipping measurements")
            return ({'status': 'python_invalid'}, conda_env)
        
        # Get test commands
        efficiency_tests = instance['efficiency_test']
        
        if not efficiency_tests:
            print(f"  ⚠️  No efficiency tests found!")
            return ({'status': 'no_tests'}, conda_env)
        
        print(f"  🧪 Found {len(efficiency_tests)} efficiency tests")
        
        # Initialize collector
        collector = MetricsCollector(
            instance_id=instance['instance_id'],
            country_code=self.country_code
        )
        
        # Measure baseline
        baseline = collector.measure_baseline(
            duration=self.config['measurement']['baseline_duration_sec']
        )
        
        # Measure each test (with individual error handling)
        all_test_results = []
        successful_tests = 0
        failed_tests = 0
        
        for i, test_name in enumerate(efficiency_tests):
            print(f"\n  📝 Test {i+1}/{len(efficiency_tests)}: {test_name}")
            
            # Verify python still exists before each test
            if not Path(python_path).exists():
                print(f"    ⚠️ Python disappeared: {python_path}")
                failed_tests += 1
                all_test_results.append({
                    'test_name': test_name,
                    'status': 'failed',
                    'error': 'Python environment was removed'
                })
                continue
            
            result = self.measure_single_test(
                collector=collector,
                test_name=test_name,
                repo_path=repo_path,
                python_path=python_path,
                repetitions=self.config['measurement']['repetitions']
            )
            
            if result and result.get('status') == 'success':
                successful_tests += 1
            else:
                failed_tests += 1
            
            all_test_results.append(result)
        
        print(f"\n  📊 Test summary: {successful_tests} passed, {failed_tests} failed")
        
        # Combine results
        results = {
            'commit': commit,
            'commit_type': commit_type,
            'baseline': baseline,
            'tests': all_test_results,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'status': 'success' if successful_tests > 0 else 'all_tests_failed',
            'repo_path': str(repo_path)  # Keep for cleanup
        }
        
        # NOTE: Don't cleanup here - let measure_instance handle it
        return (results, conda_env)
    
    def measure_instance(self, instance_id: str, output_dir: str = "data/raw/measurements"):
        """Measure a single SWE-Perf instance."""
        print("=" * 60)
        print(f"🎯 MEASURING INSTANCE: {instance_id}")
        print("=" * 60)
        
        # Get instance
        instance = self.get_instance(instance_id)
        if instance is None:
            print(f"❌ Instance '{instance_id}' not found in dataset!")
            return None
        
        # Determine Python version to use
        python_exec = get_python_executable(instance['repo'], instance['version'])
        use_conda = should_use_conda(instance['repo'], instance['version'])
        
        print(f"\n📊 Instance info:")
        print(f"  Repository: {instance['repo']}")
        print(f"  Version: {instance['version']}")
        print(f"  Python: {python_exec} ({'conda' if use_conda else 'venv'})")
        print(f"  Base commit: {instance['base_commit'][:8]}")
        print(f"  Head commit: {instance['head_commit'][:8]}")
        print(f"  Efficiency tests: {len(instance['efficiency_test'])}")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Track conda envs for cleanup
        conda_envs_to_cleanup = []
        repo_paths_to_cleanup = []
        
        try:
            # Measure base commit
            base_results, base_conda_env = self.measure_commit(
                instance=instance,
                commit=instance['base_commit'],
                commit_type='base',
                temp_dir=temp_path
            )
            if base_conda_env:
                conda_envs_to_cleanup.append(base_conda_env)
            if 'repo_path' in base_results:
                repo_paths_to_cleanup.append(base_results.pop('repo_path'))
            
            # Measure head commit
            head_results, head_conda_env = self.measure_commit(
                instance=instance,
                commit=instance['head_commit'],
                commit_type='head',
                temp_dir=temp_path
            )
            if head_conda_env:
                conda_envs_to_cleanup.append(head_conda_env)
            if 'repo_path' in head_results:
                repo_paths_to_cleanup.append(head_results.pop('repo_path'))
                
        finally:
            # Cleanup repos
            for repo_path in repo_paths_to_cleanup:
                shutil.rmtree(repo_path, ignore_errors=True)
            
            # Cleanup temp dir
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Cleanup conda envs LAST
            for env_name in conda_envs_to_cleanup:
                self.cleanup_conda_env(env_name)
        
        # Check if we have any valid results
        base_ok = base_results.get('status') == 'success'
        head_ok = head_results.get('status') == 'success'
        
        if not base_ok and not head_ok:
            print(f"\n❌ Both commits failed - not saving results")
            return None
        
        # Combine all results
        final_results = {
            'instance_id': instance_id,
            'repo': instance['repo'],
            'base_commit': instance['base_commit'],
            'head_commit': instance['head_commit'],
            'python_version': python_exec,
            'use_conda': use_conda,
            'base_measurements': base_results,
            'head_measurements': head_results,
            'original_duration_changes': instance['duration_changes'],
            'measurement_status': {
                'base_ok': base_ok,
                'head_ok': head_ok,
                'total_tests': len(instance['efficiency_test']),
                'base_successful': base_results.get('successful_tests', 0),
                'head_successful': head_results.get('successful_tests', 0)
            }
        }
        
        # Save results
        output_path = Path(output_dir) / instance_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / "measurements.json"
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print("\n" + "=" * 60)
        print(f"✅ MEASUREMENT COMPLETE!")
        print(f"  Base: {base_results.get('successful_tests', 0)} tests passed")
        print(f"  Head: {head_results.get('successful_tests', 0)} tests passed")
        print(f"💾 Results saved to: {output_file}")
        print("=" * 60)
        
        return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Measure a single SWE-Perf instance with green metrics"
    )
    parser.add_argument(
        '--instance',
        type=str,
        required=True,
        help='Instance ID to measure (e.g., astropy__astropy-16065)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/original/swe_perf_original_20251124.json',
        help='Path to SWE-Perf dataset JSON'
    )
    parser.add_argument(
        '--country',
        type=str,
        default=None,
        help='ISO country code for carbon intensity (e.g., ESP, ITA)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/measurements',
        help='Output directory for measurements'
    )
    
    args = parser.parse_args()
    
    # Create measurer
    measurer = SWEPerfMeasurer(
        dataset_path=args.dataset,
        country_code=args.country
    )
    
    # Measure instance
    measurer.measure_instance(
        instance_id=args.instance,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()