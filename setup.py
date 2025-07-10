from setuptools import setup, find_packages

setup(
    name='federal_project',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},  # 👈 告诉 setuptools：模块代码在 src 目录下
)
