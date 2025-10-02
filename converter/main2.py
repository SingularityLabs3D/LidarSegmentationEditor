import sys
from converter import run_conversion_pipeline
path = "/work/" + sys.argv[1]
run_conversion_pipeline(path, "/work/")