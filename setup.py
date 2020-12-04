from setuptools import setup, find_packages
from embryo.version import VERSION

setup(
    version=VERSION,
    author='Minhui Li',
    author_email='l.minhui@qq.com',
    description='A Library for Deep Reinforcement Learning',
    long_description=open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    python_requires='>=3.5',
    packages=find_packages(),
)