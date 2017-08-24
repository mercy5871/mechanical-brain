from distutils.core import setup
files = ["resource/*"]
setup(
        name="QPhantom-core",
        version="0.10",
        packages = [
            "QPhantom",
            "QPhantom.core",
            "QPhantom.core.metrics",
            "QPhantom.core.utils",
            "QPhantom.core.notify",
            "QPhantom.core.quant",
            "QPhantom.core.quant.feature",
            "QPhantom.core.quant.label",
            "QPhantom.core.data",
            "QPhantom.core.data.tf",
            "QPhantom.core.preprocessing",
            "QPhantom.exec",
            "QPhantom.net"
        ],
        py_modules="QPhantom",
        author = "Q-Phantom Team",
        description = "Q-Phantom core lib",
        url = "http://git.q-phantom.com/QPhantom/core",
        package_data = {'package' : files },
        install_requires=[
        ]

        )
