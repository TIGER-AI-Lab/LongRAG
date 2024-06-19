#!/bin/sh
set -e

WIKI_DUMP_FILE_IN=$1
WIKI_DUMP_FILE_OUT=${WIKI_DUMP_FILE_IN%%.*}.txt

# clone the WikiExtractor repository
# git clone https://github.com/attardi/wikiextractor.git

# extract and clean the chosen Wikipedia dump
echo "Extracting and cleaning $WIKI_DUMP_FILE_IN to $WIKI_DUMP_FILE_OUT..."
python3 -m wikiextractor.WikiExtractor  $WIKI_DUMP_FILE_IN -a
echo "Successfully extracted and cleaned $WIKI_DUMP_FILE_IN to $WIKI_DUMP_FILE_OUT"

# conda env: py34
# python3 -m wikiextractor.WikiExtractor enwiki-latest-pages-articles1.xml-p1p41242.bz2 --json -l -c -q
