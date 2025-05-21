import setuptools
from setuptools.command.egg_info import egg_info


class CustomCommand(egg_info):
    def run(self):
        import os

        os.system("make protos")
        egg_info.run(self)


setuptools.setup(
    name="orby-protos",
    version="0.0.1",
    cmdclass={
        "egg_info": CustomCommand,
    },
    packages=[
        "pb",
        "pb.v1alpha1",
        "pb.v1alpha2",
        "pb.orby_internal",
        "google.cloud.documentai.v1",
        "google.api",
        "google.type",
        "google.rpc",
        "fm",
        "common",
    ],
)
