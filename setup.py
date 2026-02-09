from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
readme = (ROOT / "README.md").read_text(encoding="utf-8")
license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]

setup(
    name="adamacs-ingest",
    version="0.1.0",
    description="Ingestion and DataJoint pipeline tooling for ADAMACS.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Daniel Muller-Komorowska",
    author_email="danielmuellermsc@gmail.com",
    url="https://github.com/SFB1089/adamacs_ingest.git",
    license=license_text,
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10",
)
