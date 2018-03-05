#!/usr/bin/python3

import os
import sys
import zipfile
import urllib.request


# create data dir if not exist
if not os.path.exists("data"):
	print("[info] create 'data' dir")
	os.makedirs("data")


def load(release, filename, dstdir):
	print("[info] loading '%s' file ..." % filename)

	url = "https://github.com/nthend/neural/releases/download/%s/%s" % (release, filename)
	print("[info] " + url)

	def dlhook(count, blockSize, totalSize):
		part = count*blockSize/totalSize
		pblen = 50
		fill = round(pblen*part)
		sys.stdout.write("\r[" + "#"*fill + " "*(pblen-fill) + ("] %d%%" % round(100*part)))
		sys.stdout.flush()

	urllib.request.urlretrieve(url, "%s/%s" % (dstdir, filename), reporthook=dlhook)
	print()

	print("[ ok ] file successfully loaded")

def extract(fpath, dstdir):
	print("[ ok ] extract '%s' file" % fpath)
	zip_ = zipfile.ZipFile(fpath, "r")
	zip_.extractall(dstdir)
	zip_.close()

# all files to download
targets = ["digits", "books"]

for target in targets:
	load(target, target + ".zip", "data")

for target in targets:
	extract("data/" + target + ".zip", "data")
