import os

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            return line.split("'")[1]

    raise RuntimeError('Unable to find version string.')


with open('requirements.txt', 'r') as requirements:
    setup(name='pseudo_attack_adversarial_suffix_generation',
          version=get_version('pseudo_attack_adversarial_suffix_generation/__init__.py'),
          install_requires=list(requirements.read().splitlines()),
          packages=find_packages(),
          description='library for creating pseudo attack adversarial suffix prompts for language models',
          python_requires='>=3.6',
          author='Ali Derakhshan, Jinhwa Kim',
          author_email='aderakh1@uci.edu',
          classifiers=[
              'Programming Language :: Python :: 3',
              'License :: OSI Approved :: MIT License',
              'Operating System :: OS Independent'
          ],
          long_description=long_description,
          long_description_content_type='text/markdown')