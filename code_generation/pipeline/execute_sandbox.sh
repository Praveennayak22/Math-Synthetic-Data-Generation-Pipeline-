#!/bin/bash
# execute_sandbox.sh — The Inspector (Step 4 of Team 2 Code Pipeline)
#
# Walks every trial folder under SANDBOX_DIR, runs run_test.sh in each,
# and prints a Pass/Fail/Error summary with pytest tail on failures.
#
# Usage:
#   bash execute_sandbox.sh [sandbox_dir]
#   Default: demo/test_run/sandbox

SANDBOX_DIR="${1:-demo/test_run/sandbox}"

if [ ! -d "$SANDBOX_DIR" ]; then
    echo "Error: Sandbox directory not found: $SANDBOX_DIR"
    exit 1
fi

echo "================================================"
echo "Execution Sandbox: $SANDBOX_DIR"
echo "================================================"

pass_count=0; fail_count=0; error_count=0

while IFS= read -r script; do
    dir=$(dirname "$script")
    echo "── $(basename "$(dirname "$dir")") / $(basename "$dir") ──"

    pushd "$dir" > /dev/null
    bash run_test.sh

    if [ -f "test_result.txt" ]; then
        result=$(cat test_result.txt)
        echo "  Result: $result"
        if [ "$result" = "Pass" ]; then
            ((pass_count++))
        else
            ((fail_count++))
            # Print last 5 lines of pytest output for quick debugging
            if [ -f "test_log.log" ]; then
                echo "  --- Error ---"
                tail -5 test_log.log | sed 's/^/  /'
            fi
        fi
    else
        echo "  Error: Missing test_result.txt"
        ((error_count++))
    fi
    popd > /dev/null

done < <(find "$SANDBOX_DIR" -name "run_test.sh" | sort)

total=$((pass_count + fail_count + error_count))
echo "================================================"
printf "Complete! Pass: %d | Fail: %d | Error: %d" $pass_count $fail_count $error_count
if [ $total -gt 0 ]; then
    printf " | Pass%%: %d%%\n" $((pass_count * 100 / total))
else
    printf "\n"
fi
echo "================================================"