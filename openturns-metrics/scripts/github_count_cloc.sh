#!/bin/bash
#
# Example
# -------
# $ bash github_count_cloc.sh > github_count_cloc.txt

# Exit immediately if a command exits with a non-zero status
set -e

# 1. Download the local cloc script if it does not exist
CLOC_VERSION="2.00" 
if [ ! -f "./cloc" ]; then
    echo "Downloading cloc v${CLOC_VERSION}..."
    curl -L "https://github.com/AlDanial/cloc/releases/download/v${CLOC_VERSION}/cloc-${CLOC_VERSION}.pl" -o cloc
    chmod +x cloc
fi

# 2. List of target URLs (cleaned from the contributor suffix)
REPOS=(
    "https://github.com/SGpp/SGpp"
    "https://github.com/cossan-working-group/OpenCossan"
    "https://github.com/UCL-CCS/EasyVVUQ"
    "https://github.com/snl-dakota/dakota"
    "https://github.com/lanl/GPMSA"
    "https://github.com/openturns/openturns"
    "https://github.com/llnl/psuade"
    "https://github.com/libqueso/queso"
    "https://github.com/idaholab/raven"
    "https://github.com/SURGroup/UQpy"
    "https://github.com/jonathf/chaospy"
    "https://github.com/sandialabs/UQTk"
    "https://github.com/cran/sensitivity"
    "https://github.com/SALib/SALib"
    "https://github.com/SMTorg/smt"
    "https://github.com/anthony-nouy/ApproximationToolbox"
    "https://github.com/anthony-nouy/tensap"
    "https://github.com/SAFEtoolbox/SAFE-python"
    "https://github.com/cran/moko"
    "https://github.com/deel-ai/puncc"
    "https://github.com/dfm/celerite"
    "https://github.com/dfm/celerite2"
    "https://github.com/dfm/george"
    "https://github.com/EmuKit/emukit"
    "https://github.com/JuliaUQ/UncertaintyQuantification.jl"
    "https://github.com/GeoStat-Framework/GSTools"
    "https://github.com/GeoStat-Framework/PyKrige"
    "https://github.com/google/neural-tangents"
    "https://github.com/GPflow/GPflow"
    "https://github.com/gpmp-dev/gpmp"
    "https://github.com/gpstuff-dev/gpstuff"
    "https://github.com/katzfuss-group/GPvecchia"
    "https://github.com/libKriging/libKriging"
    "https://github.com/mlysy/SuperGauss"
    "https://github.com/mk306/pyGPs"
    "https://github.com/openturns/otbenchmark"
    "https://github.com/relf/egobox"
    "https://github.com/ppgaluzio/MOBOpt"
    "https://github.com/secondmind-labs/trieste"
    "https://github.com/SheffieldML/GPy"
    "https://github.com/simetenn/uncertainpy"
    "https://github.com/STOR-i/GaussianProcesses.jl"
    "https://github.com/thomaspinder/GPJax"
)

# Define directories to ignore specifically for Queso
QUESO_EXCLUDES="validationCycle,validationCycle2,gpmsaTower,t01_valid_cycle,t04_bimodal,test_Regression,test_gpmsa"

# Create a temporary directory for cloning
TEMP_DIR="tmp_repos"
mkdir -p "$TEMP_DIR"

echo -e "\nStarting Line Count Analysis\n"
echo "========================================"

# 3. Loop through each repository
for repo in "${REPOS[@]}"; do
    # Extract the repository name from the URL
    repo_name=$(basename "$repo")
    
    echo "Processing: $repo_name..."
    
    # Clone the repository with depth 1
    echo "Git clone..."
    if git clone --depth 1 "$repo" "$TEMP_DIR/$repo_name" --quiet; then
        
        # Run cloc and print the summary for this repository if clone succeeds
        echo "Run cloc..."
        echo "----------------------------------------"
        echo "Results for $repo_name:"
        
        # Check if the current repository is queso
        if [ "$repo_name" = "queso" ]; then
            ./cloc "$TEMP_DIR/$repo_name" --quiet --exclude-dir="$QUESO_EXCLUDES"
        else
            ./cloc "$TEMP_DIR/$repo_name" --quiet
        fi
        
        echo "========================================"
        
        # Remove the cloned repository to save disk space
        # L'opérateur || empêche set -e d'interrompre le script en cas d'échec de rm
        rm -rf "$TEMP_DIR/$repo_name" || echo "Warning: Unable to fully remove $TEMP_DIR/$repo_name"
    else
        # Handle the failure case gracefully
        echo "Warning: Failed to clone repository from URL: $repo"
        echo "Skipping $repo_name and proceeding to the next repository."
        echo "========================================"
    fi
done

# Clean up the temporary directory
rm -rf "$TEMP_DIR" || true
echo "Analysis complete."