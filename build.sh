#!/bin/bash

cd spark-rapids-23.06.0 # 자신의 세팅에 맞는 디렉토리로 변경 
sudo mvn clean install -Dbuildver=323 -DskipTests -Drat.skip=true -DallowConventionalDistJar=true
