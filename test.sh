#!/usr/bin/bash

seed=5

output="test-seed5-v1.fits"
log="test-seed5-v1.log"
time python psfws_example_star_class.py --seed ${seed} --output ${output} &> ${log}

output="test-seed5-v2.fits"
log="test-seed5-v2.log"
time python psfws_example_star_class.py --seed ${seed} --output ${output} &> ${log}
