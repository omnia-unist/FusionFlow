#!/bin/bash
free -m
sudo sync && echo 3 > /proc/sys/vm/drop_caches
free -m
