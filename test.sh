#!/usr/bin/bash

seed=125

output="test-h.fits"
log="test-h.log"
# time python psfws_example_star_class.py --seed ${seed} --output ${output} &> ${log}
time python psfws_example_star_class.py --seed ${seed} --output ${output} --fast --ntrial 10

# output="test-seed5-v2.fits"
# log="test-seed5-v2.log"
# time python psfws_example_star_class.py --seed ${seed} --output ${output} &> ${log}
