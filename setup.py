from setuptools import setup, find_packages
 
setup(name='survival',
      version='0.0.8',
      url='https://github.com/ryu577/survival',
      license='MIT',
      author='Rohit Pandey',
      author_email='rohitpandey576@gmail.com',
      description='Add static script_dir() method to Path',
      packages=find_packages(exclude=['tests']),
      long_description=open('README.md').read(),
      zip_safe=False)

