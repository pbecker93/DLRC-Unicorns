from setuptools import setup, find_packages

setup(
    name="UnicornRobotControl",
    version='0.0.0',
    author='The Unicorns',
    packages=['camera',
              'fcn',
              'mqtt_master',
              'policy',
              'tracking',
              'color_cluster'],
    package_dir={'': 'src'},
    install_requires=['numpy', 'scipy', 'tensorflow', 'chili_tag_detector']
)