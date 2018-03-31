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
	if os.path.exists("%s/%s" % (dstdir, filename)):
		print("[ ok ] file '%s' already exists" % filename)
		return

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
targets = [
	("digits", "digits.zip"),
	("books", "books.zip"),
	("charts", "charts_v02.zip")
]

for release, filename in targets:
	load(release, filename, "data")

for release, filename in targets:
	extract("data/" + filename, "data")
