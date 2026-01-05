"""
Test astropy instance WITHOUT EnergiBridge - just run pytest directly.
"""
import subprocess
import tempfile
import shutil
from pathlib import Path

CONDA_PATH = "/opt/miniconda3/bin/conda"

def test_astropy():
    # Astropy instance info
    repo = "astropy/astropy"
    commit = "7eac388c"  # head commit
    test_name = "astropy/units/tests/test_quantity_decorator.py::test_kwarg_default[7-1]"
    
    print("=" * 60)
    print("🧪 Testing astropy WITHOUT EnergiBridge")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        repo_path = temp_path / f"astropy_{commit}"
        
        # Clone
        print(f"\n📦 Cloning {repo}...")
        subprocess.run(
            ['git', 'clone', f'https://github.com/{repo}.git', str(repo_path)],
            check=True
        )
        
        # Checkout
        print(f"🔀 Checking out {commit}...")
        subprocess.run(['git', 'fetch', 'origin', commit], cwd=repo_path, check=False)
        subprocess.run(['git', 'checkout', commit], cwd=repo_path, check=True)
        
        # Create conda env
        env_name = f"astropy_test_{commit[:8]}"
        print(f"\n📦 Creating conda env {env_name}...")
        
        subprocess.run(
            [CONDA_PATH, 'create', '-n', env_name, 'python=3.10', '-y'],
            check=True
        )
        
        # Get python path
        conda_python = f"/home/giacomo/.conda/envs/{env_name}/bin/python"
        conda_pip = f"/home/giacomo/.conda/envs/{env_name}/bin/pip"
        
        # Install setuptools
        print("📦 Installing setuptools...")
        subprocess.run([conda_pip, 'install', 'setuptools==68.0.0', 'wheel'], check=True)
        
        # Fix pyproject.toml
        print("🔧 Fixing pyproject.toml...")
        subprocess.run(
            "sed -i 's/requires = \\[\"setuptools\",/requires = \\[\"setuptools==68.0.0\",/' pyproject.toml",
            shell=True, cwd=repo_path
        )
        
        # Install astropy
        print("📦 Installing astropy (this takes a while)...")
        result = subprocess.run(
            [conda_pip, 'install', '-e', '.[test]'],
            cwd=repo_path,
            timeout=900
        )
        
        if result.returncode != 0:
            print("❌ Installation failed!")
            subprocess.run([CONDA_PATH, 'env', 'remove', '-n', env_name, '-y'])
            return
        
        print("\n✅ Installation complete!")
        
        # Run test DIRECTLY (no EnergiBridge)
        print(f"\n🧪 Running test: {test_name}")
        print("-" * 60)
        
        result = subprocess.run(
            f"cd {repo_path} && {conda_python} -m pytest '{test_name}' -v",
            shell=True,
            capture_output=False  # Show output directly
        )
        
        print("-" * 60)
        if result.returncode == 0:
            print("✅ TEST PASSED!")
        else:
            print(f"❌ TEST FAILED (return code: {result.returncode})")
        
        # Cleanup
        print(f"\n🧹 Cleaning up conda env {env_name}...")
        subprocess.run([CONDA_PATH, 'env', 'remove', '-n', env_name, '-y'])


if __name__ == "__main__":
    test_astropy()
