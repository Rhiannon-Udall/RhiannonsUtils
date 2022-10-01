#!/usr/bin/env python

# import glob
# import os
# import shutil

import setuptools
import versioneer

# Methods for getting files - unneeded at present
# files = glob.glob())
# tdir = "utils_lib"
# if os.path.exists(tdir) is False:
#     os.mkdir(tdir)
# for file in files:
#     dst = os.path.join(tdir, os.path.basename(file))
#     shutil.copy(file, dst)

if __name__ == "__main__":
    setuptools.setup(
        package_data={},
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
    )
