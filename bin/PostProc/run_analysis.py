import subprocess
import sys
import os
import re

# Absolute path to the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name):
    script_path = os.path.join(BASE_DIR, script_name)
    print(f"Running {script_name}...")
    result = subprocess.run(
        ['python', script_path],
        capture_output=True,
        text=True,
        cwd=BASE_DIR  # Ensures correct working directory
    )
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
        sys.exit(1)
    print(result.stdout)


# Run Energy.py first
run_script('Energy.py')

# Read equilibrium indices
try:
    eq_file = os.path.join(BASE_DIR, 'eq_indices.txt')
    with open(eq_file, 'r') as f:
        eq_spectra_index = int(f.readline().strip())
        eq_flux_index = int(f.readline().strip())

    # Update the scripts
    for filename, replace_val in [
        ('flux.py', f'dwn = {eq_flux_index}'),
        ('spectra.py', f'dwn = {eq_spectra_index}'),
        ('transfer.py', f'dwn = {eq_flux_index}')
    ]:
        filepath = os.path.join(BASE_DIR, filename)
        with open(filepath, 'r') as f:
            code = f.read()
        # Replace any line starting with dwn =
        code = re.sub(r'dwn\s*=\s*.*', replace_val, code, count=1)
        with open(filepath, 'w') as f:
            f.write(code)

    # Run the other scripts
    run_script('flux.py')
    run_script('spectra.py')
    run_script('transfer.py')

except FileNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)
