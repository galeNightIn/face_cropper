from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


def get_version():
    version = {}
    with open("version.py") as fp:
        exec(fp.read(), version)
    return version['__version__']


def get_requirements():
    with open("requirements.txt") as fp:
        return [line for line in fp if not line.startswith('--extra-index-url')]


setup(
    name='face_cropper',
    version=get_version(),
    packages=find_packages(),
    package_data={'face_cropper': ['models/*.caffemodel', 'models/*.pb', 'models/*.pbtxt', 'models/*.prototxt']},
    url='https://github.com/galeNightIn/face_cropper',
    description='Face cropper module',
    long_description=readme(),
    install_requires=get_requirements(),
    zip_safe=False
)
