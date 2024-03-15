#!/bin/bash

sudo cp spark-rapids-23.06.0/dist/target/rapids-4-spark_2.12-23.06.0-cuda11.jar /opt/sparkRapidsPlugin/ # 이건 경환의 실행 조건에 맞게 세팅해둔 것. submit간 extraPath의 위치에 맞게 cp 해주면 됨
sudo scp spark-rapids-23.06.0/dist/target/rapids-4-spark_2.12-23.06.0-cuda11.jar lassjupiter0@163.239.199.220:/opt/sparkRapidsPlugin/ # worker node에 전송
