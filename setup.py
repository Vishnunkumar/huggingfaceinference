import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
  'transformers',
  'torch',
]

setuptools.setup(
    name="huggingfaceinference",
    version="0.7",
    author="Vishnu Nandakumar",
    author_email="nkumarvishnu25@gmail.com",
    description="Simple inference usecases using hugging transformers library on varied use cases of the AI/ML Domain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = 'https://github.com/Vishnunkumar/huggingfaceinference/',
    packages=[
        'huggingfaceinference',
    ],
    package_dir={'huggingfaceinference': 'huggingfaceinference'},
    package_data={
        'huggingfaceinference': ['huggingfaceinference/*.py']
    },
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='huggingfaceinference',
    classifiers=(
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    ),
)
