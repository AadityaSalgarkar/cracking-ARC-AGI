# Test Suite

This directory contains comprehensive tests for the transformer implementation.

## Running Tests

### Run All Tests
```bash
uv run pytest tests/ -v
```

### Run Tests Excluding Visualization
```bash
uv run pytest tests/ -v -m "not visualization"
```

### Run Only Visualization Tests
```bash
uv run pytest tests/ -v -m "visualization"
```

### Run Specific Test File
```bash
uv run pytest tests/test_encoding_module.py -v
uv run pytest tests/test_coordinate_transformations.py -v
```

## Test Organization

### Core Functionality Tests
- `test_encoding_module.py` - Tests for the EncodingModule implementation (27 tests)
- `test_coordinate_transformations.py` - Tests for coordinate transformation compliance (12 tests)

### Visualization Tests
- `test_puzzle_visualization.py` - Tests that include matplotlib visualization (6 tests, 2 marked as visualization)

## Test Markers

- `@pytest.mark.visualization` - Tests that perform matplotlib visualization
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.integration` - Integration tests

## Notes

- Visualization tests are automatically excluded from regular test runs unless specifically requested
- All tests pass in headless environments by gracefully handling visualization failures
- Tests use fixed seeds for reproducibility