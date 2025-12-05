"""Run comparisons using analysis_lib/example_usage helpers.

Adjust the `RUN_A` and `RUN_B` paths if necessary.
"""
from example_usage import example_compare_runs, example_compare_tb

RUN_A = r'results/run41'
RUN_B = r'results/run41_1'

if __name__ == '__main__':
    print('Running CSV-based comparison...')
    try:
        example_compare_runs(RUN_A, RUN_B, out_dir='out_compare')
    except Exception as e:
        print('CSV compare failed:', e)

    print('\nRunning TensorBoard comparison (if event files exist)...')
    try:
        example_compare_tb(RUN_A, RUN_B, out_dir='out_tb_compare')
    except Exception as e:
        print('TB compare failed:', e)

    print('\nDone')
