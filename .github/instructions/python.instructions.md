---
description: 'Python coding conventions and guidelines'
applyTo: '**/*.py'
---

# Python Coding Conventions

## Python Instructions

- Write clear and concise comments for each public function and class.
- Ensure functions have descriptive names and include type hints. The return type can be omitted if it is deducible by static analysis tools.
- Provide docstrings following Google style for all public functions and classes.
- Use f-strings for string formatting (e.g., `f"Value: {value}"`).
- Break down complex functions into smaller, more manageable functions.
- Object-oriented programming principles should be followed where applicable. Avoid leaking class-specific details outside of classes. Especially, avoid env-specific logic outside of environment classes.

## General Instructions

- Always prioritize readability and clarity.
- For algorithm-related code, include explanations of the approach used.
- Write code with good maintainability practices, including comments on why certain design decisions were made.
- Handle edge cases and write clear exception handling. All exceptions should have informative messages.
- Use consistent naming conventions and follow language-specific best practices.
- Write concise, efficient, and idiomatic code that is also easily understandable.

## Code Style and Formatting

- Follow the Google style guide for Python.
- Use the style defined in `pyproject.toml` for formatting (e.g., line length, indentation).
- Ensure consistent use of whitespace and blank lines to improve readability.

## Edge Cases and Testing

- Always include test cases for critical paths of the application.
- Account for common edge cases like empty inputs, invalid data types, and large datasets.
- Include comments for edge cases and the expected behavior in those cases.
- Write unit tests for functions and document them with docstrings explaining the test cases.

## Example of Proper Documentation

```python
def calculate_area(radius: float) -> float:
    """
    Calculate the area of a circle given the radius.
    
    Args:
      radius (float): The radius of the circle.
    
    Returns:
      float: The area of the circle, calculated as π * radius^2.
    """
    import math
    return math.pi * radius ** 2
```