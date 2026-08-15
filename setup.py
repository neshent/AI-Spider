from setuptools import find_packages, setup

setup(
    name="lei",
    version="0.1.0",
    description="Lei — Reasoning Agent: a modular reference implementation of the AI agent architecture.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    extras_require={
        "web":       ["flask>=2.0.0"],
        "anthropic": ["anthropic>=0.34.0"],
        "openai":    ["openai>=1.0.0"],
        "google":    ["google-generativeai>=0.5.0"],
        "hf":        ["transformers>=4.30.0", "torch>=2.0.0", "requests>=2.28.0"],
        "test":      ["pytest>=7.0.0"],
        "all":       [
            "flask>=2.0.0",
            "anthropic>=0.34.0",
            "openai>=1.0.0",
            "google-generativeai>=0.5.0",
            "transformers>=4.30.0",
            "torch>=2.0.0",
            "requests>=2.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lei=cli:main",
        ],
    },
)
