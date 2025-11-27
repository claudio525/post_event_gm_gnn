#!/bin/zsh
#
# Script to generate and move various reports using Quarto
# Usage: ./run_reports.sh <wdata> <gnn_dir>

# Enable zsh error handling
setopt ERR_EXIT

# Check if required arguments are provided
if [[ $# -ne 2 ]]; then
    print "Error: Missing required arguments"
    print "Usage: $0 <wdata> <gnn_dir>"
    exit 1
fi

wdata=$1
gnn_dir=$2

# Check if results directory exists
if [[ ! -d $gnn_dir ]]; then
    print "Error: Results directory '$gnn_dir' does not exist"
    exit 1
fi

print "Generating reports..."

# Function to run a report only if it doesn't exist
generate_report() {
    local notebook_path=$1
    local notebook_name=$(basename $notebook_path .ipynb)
    local output_file="$gnn_dir/${notebook_name}.html"
    
    if [[ ! -f $output_file ]]; then
        print "Generating $notebook_name report..."
        quarto render $notebook_path --to html --execute -P wdata=$wdata -P results_dir=$gnn_dir && \
        mv $(dirname $notebook_path)/$notebook_name.html $gnn_dir
    else
        print "Skipping $notebook_name report (already exists)"
    fi
}

# Generate reports only if they don't exist
generate_report "../report_notebooks/loss_worst_report.ipynb"
generate_report "../report_notebooks/cv_agg_results.ipynb"
generate_report "../report_notebooks/ind_scenarios.ipynb"

print "All reports processing completed in $gnn_dir"
