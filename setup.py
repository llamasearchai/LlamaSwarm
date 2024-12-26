from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()
    # Remove comments and empty lines
    requirements = [r for r in requirements if not r.startswith("#") and r.strip()]

setup(
    name="llamaswarm-llamasearch",
    version="0.1.0",
    author="LlamaSearch AI",
    author_email="nikjois@llamasearch.ai",
    description="A simulator for multi-agent reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://llamasearch.ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.9b0",
            "isort>=5.9.3",
            "flake8>=3.9.2",
            "mypy>=0.910",
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
) 
# Updated in commit 5 - 2025-04-04 17:32:46

# Updated in commit 13 - 2025-04-04 17:32:46

# Updated in commit 21 - 2025-04-04 17:32:47

# Updated in commit 29 - 2025-04-04 17:32:47

# Updated in commit 5 - 2025-04-05 14:36:13

# Updated in commit 13 - 2025-04-05 14:36:13

# Updated in commit 21 - 2025-04-05 14:36:13

# Updated in commit 29 - 2025-04-05 14:36:13

# Updated in commit 5 - 2025-04-05 15:22:42

# Updated in commit 13 - 2025-04-05 15:22:43

# Updated in commit 21 - 2025-04-05 15:22:43

# Updated in commit 29 - 2025-04-05 15:22:43

# Updated in commit 5 - 2025-04-05 15:57:01
