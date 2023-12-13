#!/usr/bin/bash

seed=$(randint)
output="test-fudge0.30-$seed.fits"
log="test-fudge0.30-$seed.log"
time python psfws_example_star_class.py --seed ${seed} --output ${output} &> ${log}
