from setuptools import setup, find_packages

# Read the contents of your requirements file


def parse_requirements(filename):
    """ Load requirements from a pip requirements file """
    with open(filename, 'r') as file:
        lines = file.readlines()
    # Exclude lines that start with the specified prefixes
    lines = [line.strip() for line in lines if line.strip() and not line.startswith(
        ("#", "-e", "--extra-index-url", "--prefer-binary"))]
    return lines


requirements = parse_requirements('requirements.txt')

setup(
    name='onetrainer',
    version='0.1.0',
    description='Your package description',
    author='Your Name',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            # If you have scripts that should be available as command-line tools, list them here
            # 'script-name=onetrainer.scripts.script_module:main_function',
        ],
    },
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst', '*.md'],
        # And include any files found in the 'resources', 'embedding_templates', and 'training_presets' subdirectories of the package
        'onetrainer': ['resources/*', 'embedding_templates/*', 'training_presets/*'],
    },
    include_package_data=True,
    # Optional: if your package is tightly coupled with a specific version of Python
    python_requires='>=3.10',
    # Optional: classifiers give users and search engines more metadata about your package
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
    ],
)
