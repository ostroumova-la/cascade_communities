#!/bin/sh

#Make File
THE_CLASSPATH=
for i in `ls ../lib/*.jar`
  do
  THE_CLASSPATH=${THE_CLASSPATH}:${i}
done

javac -classpath ".:${THE_CLASSPATH}" util/*.java
javac -classpath ".:${THE_CLASSPATH}" evaluation/*.java
javac -classpath ".:${THE_CLASSPATH}" beans/*.java
javac -classpath ".:${THE_CLASSPATH}" handlers/*.java

javac -classpath ".:${THE_CLASSPATH}" diffusionBased/*.java
javac -classpath ".:${THE_CLASSPATH}" topicBased/*.java
javac -classpath ".:${THE_CLASSPATH}" propagationGenerator/*.java
javac -classpath ".:${THE_CLASSPATH}" cicm/*.java

#javac  myDANIBased/*.java
#javac  myRandBased/*.java
#javac -classpath ".:${THE_CLASSPATH}" core/*.java

