#!/bin/sh

#Make File
THE_CLASSPATH=
for i in `ls lib/*.jar`
  do
  THE_CLASSPATH=${THE_CLASSPATH}:${i}
done

javac -classpath ".:${THE_CLASSPATH}" util/*.java
javac -classpath ".:${THE_CLASSPATH}" evaluation/*.java
javac -classpath ".:${THE_CLASSPATH}" beans/*.java
javac -classpath ".:${THE_CLASSPATH}" core/*.java
javac -classpath ".:${THE_CLASSPATH}" handlers/*.java
javac  myDANIBased/*.java
javac  myRandBased/*.java

