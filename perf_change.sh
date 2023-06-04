#!/bin/bash

find . -name "perf.hist.*" -print0 | xargs -0 sed -i "s~%~~g"    
find . -name "perf.hist.*" -print0 | xargs -0 sed -i "s~ \+~,~g" 
