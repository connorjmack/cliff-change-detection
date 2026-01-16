# Pipeline Tests

Comprehensive test suite for the cliff-change-detection LiDAR processing pipeline.

## Overview

This directory contains tests for all 9 pipeline steps (0-8), covering:

- **Step 0**: Survey list generation
- **Step 1**: Survey list updates
- **Step 2**: Cropping with PDAL
- **Audit**: Cropped files QC (optional)
- **Step 3**: Beach removal (Random Forest)
- **Step 4**: Vegetation removal (CloudCompare CANUPO)
- **Step 5**: M3C2 change detection (CloudCompare)
- **Step 6**: DBSCAN clustering
- **Step 7**: Spatial gridding
- **Step 8**: Grid cleaning and hole filling

## Installation

Install test dependencies:

```bash
pip install pytest pytest-cov pytest-mock
```

Or using conda:

```bash
conda install pytest pytest-cov pytest-mock
```

## Running Tests

### Run all tests:

```bash
pytest
```

### Run tests for a specific pipeline step:

```bash
pytest test_0_make_survey_lists.py
pytest test_4_remove_beach_parallel.py
```

### Run with coverage report:

```bash
pytest --cov=../pipeline --cov-report=html
```

### Run only fast unit tests (skip integration tests):

```bash
pytest -m "not slow and not integration"
```

### Run with verbose output:

```bash
pytest -v
```

## Test Structure

### Fixtures (`conftest.py`)

Shared test fixtures provide:

- **Temporary directories** for test outputs
- **Mock LiDAR data structures** matching expected directory layouts
- **Sample LAS files** with realistic point cloud data
- **Mock external dependencies** (PDAL, CloudCompare, Random Forest models)
- **Location configurations** for different study sites

### Test Files

Each test file corresponds to a pipeline step or audit:

```
test_0_make_survey_lists.py     # Survey inventory generation
test_1_update_survey_lists.py   # Incremental updates
test_2_crop_files_parallel.py   # PDAL cropping
test_3_qc_cropped_files.py      # Cropping audit (optional)
test_4_remove_beach_parallel.py # Step 3: Beach classification
test_5_remove_veg_parallel.py   # Step 4: Vegetation removal
test_6_m3c2_parallel.py         # Step 5: M3C2 change detection
test_7_dbscan_parallel.py       # Step 6: Clustering
test_8_make_grids.py            # Step 7: Spatial gridding
test_9_clean_fill_grids.py      # Step 8: Grid post-processing
```

### Test Categories

Tests are organized into classes by functionality:

- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test interaction between components
- **End-to-end tests**: Test complete workflows (marked as `slow`)

## External Dependencies

Some tests require external tools installed on the system:

- **PDAL** (Step 2): Tests marked with `@pytest.mark.requires_pdal`
- **CloudCompare** (Steps 4-5): Tests marked with `@pytest.mark.requires_cloudcompare`

By default, tests that require these dependencies are mocked. To run integration tests with real dependencies:

```bash
pytest -m integration
```

## Test Data

Tests use:

1. **Synthetic data**: Generated fixtures for point clouds and grids
2. **Mocked external calls**: subprocess, file I/O
3. **Minimal real data**: Small LAS files created programmatically with `laspy`

## Coverage Goals

Target coverage by module:

- **Core logic**: >80% coverage
- **I/O functions**: >70% coverage
- **External tool wrappers**: Mocked, behavior verified
- **Error handling**: All exception paths tested

## Continuous Integration

These tests are designed to run in CI/CD pipelines without requiring:

- Large LiDAR datasets
- Licensed software (CloudCompare mocked)
- Specific OS/hardware

## Writing New Tests

When adding tests:

1. Use existing fixtures from `conftest.py`
2. Mock external dependencies (PDAL, CloudCompare)
3. Add appropriate markers (`@pytest.mark.slow`, `@pytest.mark.integration`)
4. Follow naming convention: `test_<function_name>_<scenario>`
5. Keep tests focused and isolated

Example:

```python
def test_function_name_success_case(sample_las_file, temp_dir):
    """Test that function_name works correctly with valid input."""
    result = function_name(sample_las_file, temp_dir)
    assert result is not None
    assert result.status == "success"
```

## Known Limitations

- Tests do not verify visual outputs (plots, figures)
- Some numerical precision is reduced for testing speed
- Full-scale parallel processing is not tested (would be too slow)
- CloudCompare outputs are mocked (not validated against real CC behavior)

## Troubleshooting

### Import errors

If you see `ModuleNotFoundError`, ensure the pipeline directory is in the Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/code/pipeline"
pytest
```

### Laspy version issues

Tests require laspy >= 2.0. Update with:

```bash
pip install --upgrade laspy
```

### Platform-specific failures

Some tests verify OS detection logic. Run on both macOS and Linux to ensure cross-platform compatibility.

## Contributing

When submitting changes to pipeline scripts:

1. Update corresponding tests
2. Ensure all tests pass: `pytest`
3. Verify coverage: `pytest --cov=../pipeline`
4. Add new tests for new functionality

## Contact

For questions about tests, see the main CLAUDE.md file or open an issue.
