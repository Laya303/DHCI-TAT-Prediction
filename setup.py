from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = [line for line in f.read().splitlines() 
                   if line and not line.startswith("#")]

setup(
    name="tat_project",
    version="0.1.0",
    author="Your Name",
    description="Pharmacy TAT analysis and modeling pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.0.0", "jupyter>=1.0.0", "ipykernel>=6.0.0"],
        "all": ["pytest>=7.0.0", "jupyter>=1.0.0", "ipykernel>=6.0.0"],
    },
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "get-eda=tat.scripts.get_eda:main",
            "get-delay-plots=tat.scripts.get_delay_plots:main",
            "prepare-dataset=tat.scripts.prepare_dataset:main",
            "train-models=tat.scripts.train_model:train_tat_models",
            "run-bottleneck-analysis=tat.scripts.run_bottleneck_analysis:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)