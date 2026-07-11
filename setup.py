from setuptools import setup, find_packages

setup(
    name="agent-harness",
    version="0.1.0",
    description="Reference implementation / test harness for the Reasoning Agent architecture",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    extras_require={
        "anthropic": ["anthropic>=0.34.0"],
        "test": ["pytest>=7.0.0"],
    },
    entry_points={
        "console_scripts": [
            "agent-harness=cli:main",
        ],
    },
)
