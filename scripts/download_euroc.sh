#!/bin/bash

mkdir -p datasets/euroc
cd datasets/euroc

links='
http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip
http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_02_easy/MH_02_easy.zip
http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_03_medium/MH_03_medium.zip
http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_04_difficult/MH_04_difficult.zip
http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_05_difficult/MH_05_difficult.zip
http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip
http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_02_medium/V1_02_medium.zip
http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_03_difficult/V1_03_difficult.zip
http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_01_easy/V2_01_easy.zip
http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_02_medium/V2_02_medium.zip
http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_03_difficult/V2_03_difficult.zip
'

for sc in ${links}
do
    wget ${sc}
done


unzip MH_01_easy.zip -d MH_01_easy
unzip MH_02_easy.zip -d MH_02_easy
unzip MH_03_medium.zip -d MH_03_medium
unzip MH_04_difficult.zip -d MH_04_difficult
unzip MH_05_difficult.zip -d MH_05_difficult
unzip V1_01_easy.zip -d V1_01_easy
unzip V1_02_medium.zip -d V1_02_medium
unzip V1_03_difficult.zip -d V1_03_difficult
unzip V2_01_easy.zip -d V2_01_easy
unzip V2_02_medium.zip -d V2_02_medium
unzip V2_03_difficult.zip -d V2_03_difficult

echo Done!
