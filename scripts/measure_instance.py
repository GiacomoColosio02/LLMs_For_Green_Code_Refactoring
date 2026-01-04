cd ~/LLMs_For_Green_Code_Refactoring

# Backup dello script attuale
cp scripts/measure_instance.py scripts/measure_instance.py.backup

# Crea la nuova versione con supporto conda
cat > scripts/measure_instance.py << 'ENDOFSCRIPT'
"""
Measure a single SWE-Perf instance with all metrics.
Optimized for compatibility with legacy scientific Python projects.
Supports multiple Python versions based on repository requirements.
Uses conda for Python 3.6 (sklearn), venv for others.
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

# Conda path
CONDA_PATH = "/opt/miniconda3/bin/conda"

# Python version mapping based on SWE-Perf constants.py
# Format: (repo_lower, version) -> (python_version, use_conda)
PYTHON_VERSION_MAP = {
    # xarray requires Python 3.10 (venv)
    ('pydata/xarray', '0.18'): ('python3.10', False),
    ('pydata/xarray', '0.19'): ('python3.10', False),
    ('pydata/xarray', '0.2'): ('python3.10', False),
    ('pydata/xarray', '0.20'): ('python3.10', False),
    ('pydata/xarray', '2022.03'): ('python3.10', False),
    ('pydata/xarray', '2022.06'): ('python3.10', False),
    ('pydata/xarray', '2022.09'): ('python3.10', False),
    ('pydata/xarray', '2023.04'): ('python3.10', False),
    ('pydata/xarray', '2023.07'): ('python3.10', False),
    ('pydata/xarray', '2024.05'): ('python3.10', False),
    # sklearn 0.21 requires Python 3.6 (conda)
    ('scikit-learn/scikit-learn', '0.21'): ('3.6', True),
    ('scikit-learn/scikit-learn', '0.2'): ('3.6', True),
    ('scikit-learn/scikit-learn', '0.20'): ('3.6', True),
    ('scikit-learn/scikit-learn', '0.22'): ('3.6', True),
}

# Repository-specific package constraints
REPO_PACKAGE_CONSTRAINTS = {
    'scikit-learn/scikit-learn': {
        'numpy': '1.19.2',      # Exact version for sklearn 0.21
        'cython': '0.29.24',    # Compatible Cython
        'setuptools': '58.0.4',
        'scipy': '1.5.2',
    },
    'pydata/xarray': {
        'numpy': '<2.0',
        'setuptools': '<70',
        'pandas': '<2.1',
    },
    'astropy/astropy': {
        'numpy': '<2.0',
        'setuptools': '<70',
    },
    'default': {
        'setuptools': '<70',
        'numpy': '<2.0',
        'matplotlib': '<3.9',
        'cython': '<3.0',
    }
}


def get_python_config(repo: str, version: str) -> Tuple[str, bool]:
    """
    Get Python version and whether to use conda.
    
    Returns:
        Tuple of (python_version, use_conda)
    """
    repo_lower = repo.lower()
    key = (repo_lower, version)
    
    if key in PYTHON_VERSION_MAP:
        return PYTHON_VERSION_MAP[key]
    
    return (DEFAULT_PYTHON, False)


def get_package_constraints(repo: str) -> Dict[str, str]:
    """Get package version constraints for a repository."""
    repo_lower = repo.lower()
    if repo_lower in REPO_PACKAGE_CONSTRAINTS:
        return REPO_PACKAGE_CONSTRAINTS[repo_lower]
    return REPO_PACKAGE_CONSTRAINTS['default']


class SWEPerfMeasurer:
    """Measure SWE-Perf instance with green metrics."""
    
    def __init__(self, dataset_path: str, country_code: Optional[str] = None):
        self.dataset_path = Path(dataset_path)
        self.country_code = country_code
        self.config = load_config()
        
        # Verify default Python
        try:
            result = subprocess.run(
                [DEFAULT_PYTHON, '--version'],
                capture_output=True, text=True, check=True
            )
            print(f"✅ Default Python: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(f"❌ {DEFAULT_PYTHON} not found!")
        
        # Check conda
        if Path(CONDA_PATH).exists():
            result = subprocess.run(
                [CONDA_PATH, '--version'],
                capture_output=True, text=True
            )
            print(f"✅ Conda: {result.stdout.strip()}")
        else:
            print("⚠️  Conda not found - sklearn 0.21 won't work")
        
        # Check additional Python versions
        for py_version in ['python3.10', 'python3.11']:
            try:
                result = subprocess.run(
                    [py_version, '--version'],
                    capture_output=True, text=True, check=True
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
        for instance in self.dataset:
            if instance['instance_id'] == instance_id:
                return instance
        return None
    
    def setup_repository(self, instance: Dict, temp_dir: Path, commit: str) -> Path:
        repo_name = instance['repo']
        repo_url = f"https://github.com/{repo_name}.git"
        repo_path = temp_dir / f"{repo_name.split('/')[-1]}_{commit[:8]}"
        
        print(f"  📦 Cloning {repo_name}...")
        subprocess.run(
            ['git', 'clone', repo_url, str(repo_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        
        print(f"  🔀 Fetching commit {commit[:8]}...")
        subprocess.run(
            ['git', 'fetch', 'origin', commit],
            cwd=repo_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )
        
        print(f"  🔀 Checking out commit {commit[:8]}...")
        subprocess.run(
            ['git', 'checkout', commit],
            cwd=repo_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        
        return repo_path
    
    def install_sklearn_conda(self, repo_path: Path, env_name: str) -> Optional[str]:
        """
        Install sklearn 0.21 using conda with Python 3.6.
        
        Returns:
            Path to conda environment python, or None if failed
        """
        constraints = get_package_constraints('scikit-learn/scikit-learn')
        
        try:
            # Create conda environment with Python 3.6
            print(f"  📦 [conda] Creating environment with Python 3.6...")
            subprocess.run(
                [CONDA_PATH, 'create', '-n', env_name, 'python=3.6', '-y'],
                check=True, timeout=300,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            
            # Get conda env path
            result = subprocess.run(
                [CONDA_PATH, 'env', 'list'],
                capture_output=True, text=True
            )
            env_path = None
            for line in result.stdout.split('\n'):
                if env_name in line:
                    parts = line.split()
                    for p in parts:
                        if '/' in p and env_name in p:
                            env_path = Path(p)
                            break
            
            if not env_path:
                # Default path
                env_path = Path.home() / '.conda' / 'envs' / env_name
            
            conda_pip = str(env_path / 'bin' / 'pip')
            conda_python = str(env_path / 'bin' / 'python')
            
            # Install build dependencies with exact versions
            print(f"  📦 [conda] Installing build dependencies...")
            subprocess.run(
                [conda_pip, 'install',
                 f"numpy=={constraints['numpy']}",
                 f"scipy=={constraints['scipy']}",
                 f"cython=={constraints['cython']}",
                 f"setuptools=={constraints['setuptools']}",
                 'pytest', 'joblib'],
                check=True, timeout=300
            )
            
            # Build sklearn
            print(f"  📦 [conda] Building sklearn...")
            subprocess.run(
                [conda_pip, 'install', '-e', '.', '--no-build-isolation'],
                cwd=repo_path, check=True, timeout=600
            )
            
            print(f"  ✅ [conda] sklearn installation complete")
            return conda_python
            
        except Exception as e:
            print(f"  ❌ [conda] Installation failed: {e}")
            # Cleanup failed env
            subprocess.run(
                [CONDA_PATH, 'env', 'remove', '-n', env_name, '-y'],
                capture_output=True
            )
            return None
    
    def install_dependencies_venv(self, repo_path: Path, repo: str, version: str) -> Optional[Path]:
        """Install using venv (for non-sklearn repos)."""
        python_config = get_python_config(repo, version)
        python_exec = python_config[0]
        constraints = get_package_constraints(repo)
        
        print(f"  📦 Creating virtual environment with {python_exec}...")
        
        venv_path = repo_path / "venv_sweperf"
        venv_pip = str(venv_path / 'bin' / 'pip')
        
        try:
            subprocess.run(
                [python_exec, '-m', 'venv', str(venv_path)],
                check=True, timeout=60
            )
            print(f"  ✅ Virtual environment created with {python_exec}")
            
            # Upgrade pip and setuptools
            setuptools_constraint = constraints.get('setuptools', '<70')
            subprocess.run(
                [venv_pip, 'install', '--upgrade', 
                 'pip', f"setuptools{setuptools_constraint}", 'wheel'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120
            )
            
            # Install build dependencies
            numpy_constraint = constraints.get('numpy', '<2.0')
            cython_constraint = constraints.get('cython', '<3.0')
            
            print(f"  📦 Installing build dependencies...")
            subprocess.run(
                [venv_pip, 'install',
                 'extension_helpers', 'setuptools_scm', 
                 f"cython{cython_constraint}", 
                 f"numpy{numpy_constraint}",
                 "scipy"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300
            )
            
            # Install package
            print(f"  📦 Installing project dependencies...")
            subprocess.run(
                [venv_pip, 'install', '-e', '.', '--no-build-isolation'],
                cwd=repo_path, check=True, timeout=600
            )
            
            # Install test dependencies
            matplotlib_constraint = constraints.get('matplotlib', '<3.9')
            subprocess.run(
                [venv_pip, 'install', 
                 'pytest', 'hypothesis', 'scipy', 'pytest-astropy', 'urllib3',
                 f"matplotlib{matplotlib_constraint}",
                 f"numpy{numpy_constraint}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180
            )
            
            print(f"  ✅ Dependencies installed")
            return venv_path
            
        except Exception as e:
            print(f"  ⚠️ Could not install dependencies: {e}")
            return None
    
    def install_dependencies(self, repo_path: Path, repo: str, version: str, commit: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Install dependencies using appropriate method.
        
        Returns:
            Tuple of (python_path, conda_env_name or None)
        """
        python_version, use_conda = get_python_config(repo, version)
        
        if use_conda:
            # Use conda for sklearn
            env_name = f"sklearn_{commit[:8]}"
            python_path = self.install_sklearn_conda(repo_path, env_name)
            if python_path:
                return (python_path, env_name)
            return (None, None)
        else:
            # Use venv for everything else
            venv_path = self.install_dependencies_venv(repo_path, repo, version)
            if venv_path:
                return (str(venv_path / 'bin' / 'python'), None)
            return (None, None)
    
    def cleanup_conda_env(self, env_name: str):
        """Remove conda environment."""
        if env_name:
            print(f"  🧹 Cleaning up conda env: {env_name}")
            subprocess.run(
                [CONDA_PATH, 'env', 'remove', '-n', env_name, '-y'],
                capture_output=True
            )
    
    def measure_single_test(
        self,
        collector: MetricsCollector,
        test_name: str,
        repo_path: Path,
        python_path: str,
        repetitions: int
    ) -> Optional[Dict]:
        try:
            test_command = f"cd {repo_path} && {python_path} -m pytest '{test_name}' -v"
            
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
    ) -> Dict:
        print(f"\n🔬 Measuring {commit_type} commit...")
        print("=" * 60)
        
        repo_path = self.setup_repository(instance, temp_dir, commit)
        
        python_path, conda_env = self.install_dependencies(
            repo_path, instance['repo'], instance['version'], commit
        )
        
        if python_path is None:
            print(f"  ⚠️ Skipping measurements - dependencies failed")
            return {'status': 'dependency_failed'}
        
        efficiency_tests = instance['efficiency_test']
        
        if not efficiency_tests:
            print(f"  ⚠️ No efficiency tests found!")
            if conda_env:
                self.cleanup_conda_env(conda_env)
            return {'status': 'no_tests'}
        
        print(f"  🧪 Found {len(efficiency_tests)} efficiency tests")
        
        collector = MetricsCollector(
            instance_id=instance['instance_id'],
            country_code=self.country_code
        )
        
        baseline = collector.measure_baseline(
            duration=self.config['measurement']['baseline_duration_sec']
        )
        
        all_test_results = []
        successful_tests = 0
        failed_tests = 0
        
        for i, test_name in enumerate(efficiency_tests):
            print(f"\n  📝 Test {i+1}/{len(efficiency_tests)}: {test_name}")
            
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
        
        results = {
            'commit': commit,
            'commit_type': commit_type,
            'baseline': baseline,
            'tests': all_test_results,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'status': 'success' if successful_tests > 0 else 'all_tests_failed'
        }
        
        # Cleanup
        shutil.rmtree(repo_path, ignore_errors=True)
        if conda_env:
            self.cleanup_conda_env(conda_env)
        
        return results
    
    def measure_instance(self, instance_id: str, output_dir: str = "data/raw/measurements"):
        print("=" * 60)
        print(f"🎯 MEASURING INSTANCE: {instance_id}")
        print("=" * 60)
        
        instance = self.get_instance(instance_id)
        if instance is None:
            print(f"❌ Instance '{instance_id}' not found in dataset!")
            return None
        
        python_version, use_conda = get_python_config(instance['repo'], instance['version'])
        
        print(f"\n📊 Instance info:")
        print(f"  Repository: {instance['repo']}")
        print(f"  Version: {instance['version']}")
        print(f"  Python: {python_version} ({'conda' if use_conda else 'venv'})")
        print(f"  Base commit: {instance['base_commit'][:8]}")
        print(f"  Head commit: {instance['head_commit'][:8]}")
        print(f"  Efficiency tests: {len(instance['efficiency_test'])}")
        
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        try:
            base_results = self.measure_commit(
                instance=instance,
                commit=instance['base_commit'],
                commit_type='base',
                temp_dir=temp_path
            )
            
            head_results = self.measure_commit(
                instance=instance,
                commit=instance['head_commit'],
                commit_type='head',
                temp_dir=temp_path
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        base_ok = base_results.get('status') == 'success'
        head_ok = head_results.get('status') == 'success'
        
        if not base_ok and not head_ok:
            print(f"\n❌ Both commits failed - not saving results")
            return None
        
        final_results = {
            'instance_id': instance_id,
            'repo': instance['repo'],
            'base_commit': instance['base_commit'],
            'head_commit': instance['head_commit'],
            'python_version': python_version,
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
        '--instance', type=str, required=True,
        help='Instance ID to measure'
    )
    parser.add_argument(
        '--dataset', type=str,
        default='data/original/swe_perf_original_20251124.json',
        help='Path to SWE-Perf dataset JSON'
    )
    parser.add_argument(
        '--country', type=str, default=None,
        help='ISO country code for carbon intensity'
    )
    parser.add_argument(
        '--output', type=str, default='data/raw/measurements',
        help='Output directory for measurements'
    )
    
    args = parser.parse_args()
    
    measurer = SWEPerfMeasurer(
        dataset_path=args.dataset,
        country_code=args.country
    )
    
    measurer.measure_instance(
        instance_id=args.instance,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
ENDOFSCRIPT

echo "✅ Script aggiornato con supporto conda per sklearn"