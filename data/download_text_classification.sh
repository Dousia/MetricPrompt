#!/bin/sh
DIR="./TextClassification"
mkdir $DIR
cd $DIR

rm -rf agnews
wget --content-disposition https://cloud.tsinghua.edu.cn/f/0fb6af2a1e6647b79098/?dl=1
tar -zxvf agnews.tar.gz
rm -rf agnews.tar.gz

rm -rf dbpedia
wget --content-disposition https://cloud.tsinghua.edu.cn/f/362d3cdaa63b4692bafb/?dl=1
tar -zxvf dbpedia.tar.gz
rm -rf dbpedia.tar.gz

rm -rf yahoo_answers_topics
wget --content-disposition https://cloud.tsinghua.edu.cn/f/79257038afaa4730a03f/?dl=1
tar -zxvf yahoo_answers_topics.tar.gz
rm -rf yahoo_answers_topics.tar.gz

cd ..
cp classes/agnews/classes.txt TextClassification/agnews/classes.txt
cp classes/dbpedia/classes.txt TextClassification/dbpedia/classes.txt
cp classes/yahoo_answers_topics/classes.txt TextClassification/yahoo_answers_topics/classes.txt

