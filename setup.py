from setuptools import setup, find_packages

setup(
    name="mini-llm-engine",
    version="0.1.0",
    description="A minimal LLM inference engine demonstrating Continuous Batching and Paged KV Cache",
    author="Your Name",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "benchmarks*", "examples*"]),
    install_requires=[
        "pytest>=7.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.5"],
        "model": ["torch>=2.0", "transformers>=4.35"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
