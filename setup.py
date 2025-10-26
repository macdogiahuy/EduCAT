from setuptools import find_packages, setup

setup(
    name='EduCAT',
    version='0.0.1',
    author='Yuting Ning',
    author_email='ningyt@mail.ustc.edu.cn',
    packages=find_packages(),
    description=""" A CAT Framework """,
    long_description_content_type="text/markdown",
    # ensure README is read using utf-8 to avoid encoding errors on Windows
    long_description=open('README.md', encoding='utf-8').read(),
    url='https://github.com/bigdata-ustc/CAT',
    install_requires=[
        'torch',
        'vegas',
        'numpy',
        'scikit-learn',
        'scipy',
    ],  # And any other dependencies foo needs
    entry_points={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
