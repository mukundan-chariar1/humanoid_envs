#!/bin/bash

# author: Mukundan Chariar
# userid: mchariar
# This script takes in a logfile, displays it using less, and can delete the logfile if specified. Created since my dual boot creates a lot of logfiles (>40gb) per week somehow, and it is annoying to view them again and again

LOG_FILE="$1"

if [ ! -f "$1" ]; then
    echo "Error: File '$1' not found"
    exit 1
fi

less "$1"

if [ "$2" = "--delete" ]; then
    read -p "Delete $1? [y/N] " answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        rm -v "$1"
    else
        echo "Not deleted"
    fi
fi