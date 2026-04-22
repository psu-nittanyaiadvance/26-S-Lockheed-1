import subprocess
import os
import sys

def main():
    print("====================================")
    print("      Z-Splat Interactive Runner     ")
    print("====================================")
    
    # 1. Ask for dataset path
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = input("\nEnter the path to a dataset (COLMAP sparse directory or dataset root): ").strip()
    
    if not dataset_path:
        print("Error: Dataset path cannot be empty.")
        sys.exit(1)
        
    if not os.path.exists(dataset_path):
        print(f"Warning: The path '{dataset_path}' does not exist on the current filesystem.")
        print("Make sure this path is correctly mounted if running in Docker.")
        
    # 2. Ask for model version
    version_valid = False
    version_choice = "v2"
    while not version_valid:
        choice = input("\nWhich version would you like to run? [v1 (Standard) / v2 (Sonar/Depth Enhanced)] (default: v2): ").strip().lower()
        if choice in ["", "v2", "2"]:
            version_choice = "v2"
            version_valid = True
        elif choice in ["v1", "1"]:
            version_choice = "v1"
            version_valid = True
        else:
            print("Invalid choice. Please enter 'v1' or 'v2'.")

    script_to_run = "train.py" if version_choice == "v1" else "train_v2.py"
    
    # 3. Formulate command
    cmd = [sys.executable, script_to_run, "-s", dataset_path]
    
    # 4. Prompt for iterations
    iterations = input("\nHow many iterations would you like to run? [Press enter for default (30000)]: ").strip()
    if iterations:
        if iterations.isdigit():
            cmd.extend(["--iterations", iterations])
        else:
            print("Warning: Iterations must be a number. Using default 30000.")

    print("\n------------------------------------")
    print(f"Running Command: {' '.join(cmd)}")
    print("------------------------------------\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining interrupted or failed with exit code {e.returncode}.")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nProcess canceled by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
