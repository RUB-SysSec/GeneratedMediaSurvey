from setuptools import find_packages, setup

setup(
    name='dfsurvey',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "flask",
        "Flask-SQLAlchemy",
        "psycopg2",
        "toml",
    ],
    tests_require=[
        "pytest",
        "beautifulsoup4",
        "selenium",
        "pytest-xdist",
        "aiohttp",
        "aiodns",
    ],
)
